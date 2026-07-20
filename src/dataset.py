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
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image'].float() / 255.0
        
        return image, label
    
def get_clinical_transform(is_train = True):
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
        
def build_dataloaders(image_dir, batch_size, val_split = 0.2):
    all_paths = []
    all_labels = []
    classes = sorted(os.listdir(image_dir))
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                all_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(label_idx)
    print(f"Total {len(all_paths)} images, {len(classes)} classes")
    
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size = val_split, stratify = all_labels, random_state = 42
    )
    print(f"train : {len(train_paths)}, val : {len(val_paths)}")
    
    
    class_counts = np.bincount(train_labels)
    
    class_weights = 1.0 / class_counts
    
    sample_weights = np.array([class_weights[t] for t in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_dataset = PressureUlcerDataset(train_paths, train_labels, transform = get_clinical_transform(is_train=True))
    val_dataset = PressureUlcerDataset(val_paths, val_labels, transform=get_clinical_transform(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = sampler)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    
    return train_loader, val_loader, classes
        
        

if __name__ == "__main__":
    t_loader, v_loader, cls_names = build_dataloaders(image_dir="./data", batch_size = 4, val_split = 0.2)
    
    images, labels = next(iter(t_loader))
    print(f"{labels}")