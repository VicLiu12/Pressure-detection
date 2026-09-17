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


# CoordAtt 非線性激活函數
class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0
    

class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, reduction = 32):
        super(CoordAttMeanMax, self).__init__()
        mip = max(8, inp // reduction)
        
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_h_max = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w_max = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
    
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h_avg = self.pool_h_avg(x)
        x_h_max = self.pool_h_max(x)
        x_h = x_h_avg + x_h_max
        
        x_w_avg = self.pool_w_avg(x)
        x_w_max = self.pool_w_max(x)
        x_w = x_w_avg + x_w_max
        
        y = torch.cat([x_h, x_w.transpose], dim = 2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)
        
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return identity * a_h * a_w
        
       
    
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
            
            self.coordatt4 = CoordAttMeanMax(2048)
            self.coordatt3 = CoordAttMeanMax(1024)
            self.coordatt2 = CoordAttMeanMax(512)
            self.coordatt1 = CoordAttMeanMax(256)
            
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
        
        c4 = self.coordatt4(self.feature_map['layer4'])
        c3 = self.coordatt3(self.feature_map['layer3'])
        c2 = self.coordatt2(self.feature_map['layer2'])
        c1 = self.coordatt1(self.feature_map['layer1'])
        
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



