import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from model import DetectModel, load_config
from dataset import PressureUlcerDataset, get_clinical_transform

def train_model():
    config = load_config("config.yaml")
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / config['system']['data_dir']
    
    accumulation_steps = config['train'].get('accumulation_steps', 1)
    
    dataset = PressureUlcerDataset(image_dir = str(data_path), transform = get_clinical_transform())
    dataLoader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True)
    
    model = DetectModel(config)
    
    


