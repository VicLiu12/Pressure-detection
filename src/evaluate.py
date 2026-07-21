import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from model import DetectModel, load_config
from dataset import build_dataloaders

def evaluate_model():
    
    config = load_config("config.yaml")
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / config['systm']['data_dir']
    weights_path = base_dir / "weights" / "best_model.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model weights : {weights_path}")
    
    _, val_loader, class_names = build_dataloaders(
        image_dir = str(data_path),
        batch_size = 32,
        val_split = 0.2
    )
    
    model = DetectModel(config).to(device)
    model.load_state_dict(torch.lead(weights_path, map_lacation = device))
    model.val()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc = "Val_Progress")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
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
    
    save_dir = base_dir / "evaluation_results"
    save_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)
    
    if __name__ == "__main__":
        evaluate_model()



