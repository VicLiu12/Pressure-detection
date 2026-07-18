import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from model import DetectModel, load_config
from dataset import PressureUlcerDataset, get_clinical_transform

def train_model():
    config = load_config("config.yaml")
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / config['system']['data_dir']
    
    accumulation_steps = config['train'].get('accumulation_steps', 1)
    
    dataset = PressureUlcerDataset(image_dir = str(data_path), transform = get_clinical_transform())
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)
    
    model = DetectModel(config)
    
    #CrossEntropyLoss 損失函數 & 優化器
    criterion = nn.CrossEntropyLoss()
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
                       


