import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path

def load_config(config_name = "config.yaml"):
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / config_name
    
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    
#尋找大部分特徵
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1) #最大池化
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias = False) #1x1的卷積核
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)  #透過sigmoid函數轉為0~1的權重
    

#尋找特定特徵
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):  #7x7的卷積核
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding = padding, bias = False)
        self.sigmiod = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv1(x)
        return self.sigmiod(x)
    
    
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio = 16, kernel_size = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
       
    
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
            
            self.backbone.fc = nn.Identity()
            
            #註冊攔截器
            self.backbone.layer1.register_forward_hook(self.get_hook('layer1'))
            self.backbone.layer2.register_forward_hook(self.get_hook('layer2'))
            self.backbone.layer3.register_forward_hook(self.get_hook('layer3'))
            self.backbone.layer4.register_forward_hook(self.get_hook('layer4'))
            
            self.cbam4 = CBAM(2048)
            self.cbam3 = CBAM(1024)
            self.cbam2 = CBAM(512)
            self.cbam1 = CBAM(256)
            
            #FPN 轉換
            self.fpn_latlayer4 = nn.Conv2d(2048, 256, kernel_size=1)
            self.fpn_latlayer3 = nn.Conv2d(1024, 256, kernel_size=1)
            self.fpn_latlayer2 = nn.Conv2d(512, 256, kernel_size=1)
            self.fpn_latlayer1 = nn.Conv2d(256, 256, kernel_size=1)
            
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            self.holographic_dim = 1024
            #特徵分類頭
            self.classifier_head = nn.Linear(self.holographic_dim, num_classes)
            
            #階層對比頭
            self.projection_head = nn.Sequential(
                nn.Linear(self.holographic_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace = True),
                nn.Linear(512, 128)
            )
        
        else :
            raise ValueError("Model ERROR")

    #取得layer1~layer4過程的特徵圖提取出來
    def get_hook(self, layer_name):
        def hook_fn(module, input, output):
            self.feature_map[layer_name] = output
        return hook_fn


    #FPN (Feature Pyramid Network)    
    def forward(self, x):      
        _ = self.backbone(x)
        
        c4 = self.cbam4(self.feature_map['layer4'])
        c3 = self.cbam3(self.feature_map['layer3'])
        c2 = self.cbam2(self.feature_map['layer2'])
        c1 = self.cbam1(self.feature_map['layer1'])
        
        p4 = self.fpn_latlayer4(c4)
        p4_upsampled = F.interpolate(p4, size = c3.shape[2:], mode = 'bilinear', align_corners = False)
        
        p3 = self.fpn_latlayer3(c3) + p4_upsampled
        p3_upsampled = F.interpolate(p3, size = c2.shape[2:], mode = 'bilinear', align_corners = False)
        
        p2 = self.fpn_latlayer2(c2) + p3_upsampled
        p2_upsampled = F.interpolate(p2, size = c1.shape[2:], mode = 'bilinear', align_corners = False)
        
        p1 = self.fpn_latlayer1(c1) + p2_upsampled
        
        fused_features = {
            'p4' : p4,
            'p3' : p3,
            'p2' : p2,
            'p1' : p1
        }
        
        pool_p4 = self.global_pool(p4).flatten(1)
        pool_p3 = self.global_pool(p3).flatten(1)
        pool_p2 = self.global_pool(p2).flatten(1)
        pool_p1 = self.global_pool(p1).flatten(1)

        
        holographic_vector = torch.cat([pool_p4, pool_p3, pool_p2, pool_p1], dim = 1)
        
        #分類結果
        classification_result = self.classifier_head(holographic_vector)
        
        projected_feature = self.projection_head(holographic_vector)
        projected_feature = F.normalize(projected_feature, p=2, dim=1)
        
        return classification_result, fused_features, projected_feature
    
if __name__ == "__main__":
    config = load_config("config.yaml")
    
    model = DetectModel(config)
    print(f"載入模型 : {config['model']['name']}")
    
    test_input = torch.randn(config['train']['batch_size'], 3, 224, 224)
    
    output_class, output_feature, output_proj = model(test_input)
    
    print(f"輸入維度 : {test_input.shape}")
    print(f"輸出維度 : {output_class.shape} (Batch Size, 類別數)")
    print(f"對比投影維度 : {output_proj.shape} (Batch Size, 空間維度)")
    print("特徵圖輸出維度 : ")
    for layer, f_map in output_feature.items():
        print(f" {layer} 維度 : {f_map.shape}")



