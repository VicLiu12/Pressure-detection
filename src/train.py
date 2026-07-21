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

from model import DetectModel, load_config
from dataset import build_dataloaders
from loss import FocalLoss
from visualize import Grad_CAM


class SAM(torch.optim.Optimizer):
    def __init__(self, parmas, base_optimizer, rho = 0.05, **kwarge):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho = rho, **kwarge)
        super(SAM, self).__init__(parmas, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwarge)
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
                
        if self.zero_grad : self.zero_grad()
        
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
                for group in self.parma_groups for p in group["params"]
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


def train_model():
    config = load_config("config.yaml")
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / config['system']['data_dir']
    
    accumulation_steps = config['train'].get('accumulation_steps', 1)
    
    #偵測使否有cuda可以使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, class_names = build_dataloaders(
        image_dir=str(data_path),
        batch_size=config['train']['batch_size'],
        val_split=0.2
    )
    
    track_images, _ = next(iter(val_loader))
    track_image = track_images[0:1].to(device)
    
    track_dir = base_dir / "history_detect_images"
    track_dir.mkdir(exist_ok=True)
    
    model = DetectModel(config).to(device)
    
    #Focal Loss 損失函數 & 優化器
    criterion = FocalLoss(alpha = 1.0, gamma = 2.0)
    optimizer = optim.Adam(model.parameters(), lr = config['train']['learning_rate'])
    
    epochs = config['train']['epochs']
    print(f"Accumulation_steps : {accumulation_steps}")
    
    scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-6)
    
    best_val_acc = 0.0
    
    patience = 10 
    early_stop = 0
    
    for epoch in range(1, epochs + 1):
        
        #Train process
        model.train()
        runnning_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        train_bar = tqdm(
            train_loader, 
            total = len(train_loader), 
            desc = f"Epoch {epoch}/{epochs} [Train, LR: {current_lr:.1e}]]"
        )
        
        for step, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 5.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
            runnning_loss += loss.item() * accumulation_steps
            train_bar.set_postfix({'Loss' : f"{runnning_loss / (step + 1):.4f}"})
        
        #Validation process 
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, total = len(val_loader), desc = f"Epoch {epoch}/{epochs} [Val]")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Accuracy of validation : {val_acc: 2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = base_dir / "weights"
            save_dir.mkdir(exist_ok = True)
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            early_stop = 0
            
        else:
            early_stop += 1
            print("Accuracy is not grow up")
            
        save_image(model, track_image, epoch, track_dir)
        
        scheduler.step()
        
        if early_stop >= patience:
            print("Early STOP")
            break
                
if __name__ == "__main__":
    train_model()
                       


