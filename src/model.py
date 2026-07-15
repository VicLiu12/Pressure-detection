import yaml
import torch
import torch.nn as nn
from torchvision import models

def load_config(config_path = "config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
class DetectModel(nn.Model):
    def __init__(self, config):
        super(DetectModel, self).__init__()
        
        model_name = config['model']['name']
        num_classes = config['system']['num_classes']
        pretrained = config['model']['pretrained']
        
        if model_name == "resnet50" :
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights = weights)
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        
        else :
            raise ValueError("Model unsupported")
        
    def forward(self, x):
        return self.backbone(x)
    
if __name__ == "__main__":
    config = load_config("../config.yaml")
    
    model = DetectModel(config)
    print(f"載入模型 : {config['model']['name']}")
    
    test_input = torch.randn(config['train']['batch_size'], 3, 224, 224)
    
    output = model(test_input)
    print(f"輸入維度 : {test_input.shape}")
    print(f"輸出維度 : {output.shape} (Batch Size, 類別數)")



