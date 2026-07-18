import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import torch.nn.functional as F

from model import DetectModel, load_config

class Grad_CAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def generate_cam(self, input_image, target_layer):
        output_class, fused_features = self.model(input_image)
        backbone_layer_name = target_layer.replace('p', 'layer')
        
        target_feature = self.model.feature_map[backbone_layer_name]
        
        target_feature.retain_grad()
        
        target_class = torch.argmax(output_class, dim = 1).item()
        score = output_class[0, target_class]
        
        self.model.zero_grad()
        score.backward(retain_graph = True)
        
        gradients = target_feature.grad[0]
        activations = target_feature[0]
        
        weights = torch.mean(gradients, dim = (1, 2))
        
        cam = torch.zeros(activations.shape[1:], dtype = torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = F.relu(cam)
        cam = cam.detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    
    


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
    
    cam_executive = Grad_CAM(model)
    
    for i, (key, title) in enumerate(zip(keys, titles)):
        cam = cam_executive.generate_cam(image_tensor, target_layer = key)
        cam_resized = cv2.resize(cam, (224, 224), interpolation = cv2.INTER_CUBIC)
        
        axes[i+1].imshow(img_np, alpha = 0.6)
        axes[i+1].imshow(cam_resized, cmap = 'jet', alpha = 0.5)
        axes[i+1].set_title(title, fontsize = 12)
        axes[i+1].axis('off')
        
    fig.tight_layout()
    plt.show()
    
    

    
if __name__ == "__main__":
    config = load_config("config.yaml")
    model = DetectModel(config)
    
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
    
    Multi_Feature_Image(real_images, model)
