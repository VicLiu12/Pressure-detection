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

from model import DetectModel, load_config
from dataset import build_dataLoaders
from loss import FocalLoss
from visualize import Grad_CAM


def save_cam(model,image_tansor, epoch, save_dir):
    cam_main = Grad_CAM(model)
    
    img_np = image_tansor[0].cpu().permute(1, 2, 0).detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    
    fig, axes = plt.subplot(1, 5, figsize = (20, 4))
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
    
    plt.savfig(save_dir / f"epoch_{epoch:03d}.png")
    plt.close(fig)


def train_model():
    config = load_config("config.yaml")
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / config['system']['data_dir']
    
    accumulation_steps = config['train'].get('accumulation_steps', 1)
    
    #偵測使否有cuda可以使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = PressureUlcerDataset(image_dir = str(data_path), transform = get_clinical_transform())
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)
    
    model = DetectModel(config)
    
    #Focal Loss 損失函數 & 優化器
    criterion = FocalLoss(alpha = 1.0, gamma = 2.0)
    optimizer = optim.Adam(model.parameter(), lr = config['train']['leaning_size'])
    
    epochs = config['train']['epochs']
    print(f"Accumulation_steps : {accumulation_steps}")
    
    for epoch in range(epochs):
        model.train()
        runnning_loss = 0.0
        
        progress_bar = tqdm(enumerate(dataloader), total = len(dataloader), desc = f"Epoch {epoch+1}/{epochs}")
        
        for step, (images, labels) in progress_bar:
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
            runnning_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({'Loss' : f"{runnning_loss / (step + 1):.4f}"})
    
if __name__ == "__main__":
    train_model()
                       


