import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class PressureUlcerDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        
        self.image_dir = image_dir
        self.transform = transform
        self.image_path = []
        self.labels = []
        
        self.classes = sorted(os.listdir(image_dir))
        
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.image_path.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

if __name__ == "__main__":
    clinical_transform = A.Compose([
        A.Resize(224,224),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 20, val_shift_limit = 10, p = 0.5),
        ToTensorV2()
    ])
    
    dataset = PressureUlcerDataset(image_dir="./data", transform=clinical_transform)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)
    
    print(f"共有 {len(dataset)} 張影像")
    print(f"有 {dataset.classes} 個分類")
    
    images, labels = next(iter(dataloader))
    
    print(f"影像 Batch 維度 (Batch, Channels, Height, Width): {images.shape}")
    print(f"標籤 Batch: {labels}")
    
    print("complete")
