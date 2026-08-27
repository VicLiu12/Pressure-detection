import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import random

from model import DetectModel, load_config
from dataset import build_dataloaders
from loss import JoinLoss
from visualize import Grad_CAM
from v_tsne import tsne_image


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho = 0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho = rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
    @torch.no_grad()
    def first_step(self, zero_grad = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"]/ (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None : continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w 
                
        if zero_grad : self.zero_grad()
        
    @torch.no_grad()
    def second_step(self, zero_grad = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None : continue
                p.sub_(self.state[p]["e_w"])
                
        self.base_optimizer.step()
        
        if zero_grad : self.zero_grad()
        
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p = 2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p = 2
        ) 
        return norm


def save_image(model,image_tansor, epoch, save_dir):
    cam_main = Grad_CAM(model)
    
    img_np = image_tansor[0].cpu().permute(1, 2, 0).detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    
    fig, axes = plt.subplots(1, 5, figsize = (20, 4))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Epoch {epoch} - Input", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    keys = ['p4', 'p3', 'p2', 'p1']
    titles = ['P4 (Macro)', 'P3 (Mid)', 'P2 (Edge)', 'P1 (Micro)']
    
    for i, (key, title) in enumerate(zip(keys, titles)):
        cam = cam_main.generate_cam(image_tansor, target_layer=key)
        cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        axes[i+1].imshow(img_np, alpha = 0.6)
        axes[i+1].imshow(cam_resized, cmap = 'jet', alpha = 0.5)
        axes[i+1].set_title(title, fontsize = 12)
        axes[i+1].axis('off')
        
    fig.tight_layout()
    
    plt.savefig(save_dir / f"epoch_{epoch:03d}.png")
    plt.close(fig)
    
def evaluate_model(model, val_loader, device, class_names, base_dir):
    print("\n")
    print("\n")
    
    weights_path = base_dir / "weights" / "best_model.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")
    print(f"Model weights : {weights_path}")
        
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Can't find best_model.pth")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc = "Val_Progress")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
                
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n")
    report = classification_report(all_labels, all_preds, target_names = class_names, digits = 4)
    print(report)
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm, 
                annot = True, fmt = "d", 
                cmap = "Blues",
                xticklabels = class_names, yticklabels = class_names
        )
    plt.title("Confusion Matrix of Pressure Ulcer Classification", fontsize=16, fontweight='bold')
    plt.ylabel("Clinical Grade", fontsize = 12)
    plt.xlabel("Predicted Grade", fontsize = 12)
    plt.tight_layout()

    save_dir = base_dir / "evaluation_results"
    save_dir.mkdir(exist_ok=True)
    cm_save_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_save_path, dpi = 400)
    plt.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model():
    config = load_config("config.yaml")
    
    seed = config['system']
    set_seed(seed)
    print(f"Seed : {seed}")
    
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / config['system']['data_dir']
    
    accumulation_steps = config['train'].get('accumulation_steps', 1)
    
    #偵測使否有cuda可以使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, class_names = build_dataloaders(
        image_dir=str(data_path),
        batch_size=config['train']['batch_size'],
        val_split=0.2,
        seed = seed
    )
    
    track_images, _ = next(iter(val_loader))
    track_image = track_images[0:1].to(device)
    
    track_dir = base_dir / "history_detect_images"
    track_dir.mkdir(exist_ok=True)
    
    model = DetectModel(config).to(device)
    
    #Focal Loss 損失函數 & 優化器
    criterion = JoinLoss(alpha = 1.0, gamma = 2.0, lambda_con = 0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr = config['train']['learning_rate'])
    
    epochs = config['train']['epochs']
    print(f"Accumulation_steps : {accumulation_steps}")
    
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), 
                    base_optimizer, 
                    lr = config['train']['learning_rate']
                )
    
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max = epochs, eta_min = 1e-6)
    
    best_val_acc = 0.0
    
    history_train_loss = []
    history_val_acc = []
    history_cls_loss = []
    history_con_loss = []
    
    for epoch in range(1, epochs + 1):
        
        #Train process
        model.train()
        runnning_loss = 0.0
        running_cls_loss = 0.0
        running_con_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        train_bar = tqdm(
            train_loader, 
            total = len(train_loader), 
            desc = f"Epoch {epoch}/{epochs} [Train, LR: {current_lr:.1e}]"
        )
        
        for step, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            class_out, _ , proj_out = model(images)
            loss, cls_loss, con_loss = criterion(class_out, proj_out, labels)
            
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                
                #梯度裁減 (範數限制在5.0以內)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5.0)
                
                optimizer.first_step(zero_grad = True)
                
                class_out_2, _, proj_out_2 = model(images)
                loss_2, _, _ = criterion(class_out_2, proj_out_2, labels)
                loss_2 = loss_2 / accumulation_steps
                loss_2.backward()
                
                optimizer.second_step(zero_grad = True)
                
            runnning_loss += loss.item() * accumulation_steps
            running_cls_loss += cls_loss.item()
            running_con_loss += con_loss.item()
            
            train_bar.set_postfix({
                'Total' : f"{runnning_loss / (step + 1):.4f}",
                'Cls' : f"{running_cls_loss / (step + 1):.4f}",
                'Con' : f"{running_con_loss / (step + 1):.4f}"
            })

        history_train_loss.append(runnning_loss / len(train_loader)) 
        history_cls_loss.append(running_cls_loss / len(train_loader))
        history_con_loss.append(running_con_loss / len(train_loader))
        
        #Validation process 
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, total = len(val_loader), desc = f"Epoch {epoch}/{epochs} [Val]")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs, _, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Accuracy of validation : {val_acc: 2f}%")
        history_val_acc.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = base_dir / "weights"
            save_dir.mkdir(exist_ok = True)
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("The best model is saved in folder weights")
            
        save_image(model, track_image, epoch, track_dir)
        scheduler.step()
            
    plt.figure(figsize = (18, 5))


    #Train Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), history_train_loss, marker = 'o', color = 'blue', label = 'Train Loss ')
    plt.title('Total Training Loss', fontsize = 14, fontweight = 'bold')
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Loss', fontsize = 12)
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.legend()
    
    #Loss Comparison curve
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs + 1), history_cls_loss, marker = '^', color = 'darkorange', label = 'Classification Loss')
    plt.plot(range(1, epochs + 1), history_con_loss, marker = 'd', color = 'purple', label = 'Ordinal SupCon Loss')
    plt.title('Loss Components Breakdown', fontsize = 14, fontweight = 'bold')
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Loss Value', fontsize = 12)
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.legend()

    #Validation Accuracy curve
    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs + 1), history_val_acc, marker = 's', color = 'green', label = 'Validation Accuracy ')
    plt.title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Accuracy (%)', fontsize = 12)
    plt.grid(True, linestyle = '--', alpha = 0.7)
    plt.legend()

    plt.tight_layout()
    eval_dir = base_dir / "evaluation_results"
    eval_dir.mkdir(exist_ok = True)
    curve_save_path = eval_dir / "learning_curve.png"
    plt.savefig(curve_save_path, dpi = 400)
    plt.close()
    
    #更新記憶體中的best_model.pth
    weights_path = base_dir / "weights" / "best_model.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location = device))
    
    evaluate_model(model, val_loader, device, class_names, base_dir)
    tsne_image(model, val_loader, device, class_names, base_dir)
        
        
if __name__ == "__main__":
    train_model()
                       


