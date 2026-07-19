import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


class PressureUlcerDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = self.image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image'].float() / 255.0
        
        return image, label
    
def get_clinical_transform(is_train = True)):
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.RandomBrightnessContrast(p = 0.5),
            A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 20, val_shift_limit = 10, p = 0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            ToTensorV2()
        ])
        
def build_dataLoaders(image_dir, batch_size, val_split = 0.2):
    all_paths = []
    all_labels = []
    classes = sorted(os.listdir(image_dir))
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                all_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(label_idx)
        
        
        

if __name__ == "__main__":
    clinical_transform = get_clinical_transform()
    
    dataset = PressureUlcerDataset(image_dir="./data", transform=clinical_transform)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)
    
    print(f"共有 {len(dataset)} 張影像")
    print(f"有 {dataset.classes} 個分類")
    
    images, labels = next(iter(dataloader))
    
    print(f"影像 Batch 維度 (Batch, Channels, Height, Width): {images.shape}")
    print(f"標籤 Batch: {labels}")
    
    print("complete")
