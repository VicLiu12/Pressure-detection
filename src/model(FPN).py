import yaml
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

def load_config(config_name = "config.yaml"):
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / config_name
    
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    
class DetectModel(nn.Module):
    def __init__(self, config):
        super(DetectModel, self).__init__()
        
        model_name = config['model']['name']
        num_classes = config['system']['num_classes']
        pretrained = config['model']['pretrained']
        
        self.feature_map = {}
        
        if model_name == "resnet50" :
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights = weights)
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
            self.backbone.layer1.register_forward_hook(self.get_hook('layer1'))
            self.backbone.layer2.register_forward_hook(self.get_hook('layer2'))
            self.backbone.layer3.register_forward_hook(self.get_hook('layer3'))
            self.backbone.layer4.register_forward_hook(self.get_hook('layer4'))
        
        else :
            raise ValueError("Model unsupported")
    
    def get_hook(self, layer_name):
        def hook_fn(module, input, output):
            self.feature_map[layer_name] = output
        return hook_fn
        
    def forward(self, x):
        classification_result = self.backbone(x)
        return classification_result, self.feature_map
    
if __name__ == "__main__":
    config = load_config("config.yaml")
    
    model = DetectModel(config)
    print(f"載入模型 : {config['model']['name']}")
    
    test_input = torch.randn(config['train']['batch_size'], 3, 224, 224)
    
    output_class, output_feature = model(test_input)
    
    print(f"輸入維度 : {test_input.shape}")
    print(f"輸出維度 : {output_class.shape} (Batch Size, 類別數)")
    print("特徵圖輸出維度 : ")
    for layer, f_map in output_feature.items():
        print(f" {layer} 維度 : {f_map.shape}")



