import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from model import DetectModel, load_config

def Multi_Feature_Image(image_tensor, feature_dicts):
    img_np = image_tensor[0].permute(1, 2, 0).detach().numpy()
    
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    
    fig, axes = plt.subplots(1, 5, figsize = (20, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Input", fontsize = 14, fontweight = 'bold')
    axes[0].axis('off')
    
    keys = ['p4', 'p3', 'p2', 'p1']
    titles = [
        'P4 (Macro / 7x7)', 
        'P3 (Mid / 14x14)', 
        'P2 (Edge / 28x28)', 
        'P1 (Micro / 56x56)'
    ]
    
    for i, (key, title) in enumerate(zip(keys, titles)):
        f_map = feature_dicts[key][0]
        
        attention = torch.mean(f_map, dim = 0).detach().numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-5)
        
        axes[i+1].imshow(img_np, alpha = 0.6)
        axes[i+1].imshow(attention, cmap = 'jet', alpha = 0.6, extent = [0, 224, 224, 0])
        axes[i+1].set_title(title, fontsize = 12)
        axes[i+1].axis('off')
        
    fig.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    config = load_config("config.yaml")
    model = DetectModel(config)
    
    model.eval()
    print("Success loading Model")
    
    from dataset import PressureUlcerDataset, get_clinical_transform
    from torch.utils.data import DataLoader
    
    clinical_transform = get_clinical_transform()
    
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / config['system']['data_dir']
    print(f"Dataset path : {data_path}")
    
    dataset = PressureUlcerDataset(image_dir = str(data_path), transform = clinical_transform)
    
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)
    
    real_images, labels = next(iter(dataloader))
    
    output_classes, fused_features = model(real_images)
    Multi_Feature_Image(real_images, fused_features)
        
