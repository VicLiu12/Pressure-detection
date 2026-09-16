import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path

def load_config(config_name="config.yaml"):
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / config_name
    
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# --------------------------------------------------------
# 引入 CoordAtt 的非線性激活函數 h_swish
# --------------------------------------------------------
class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0

# --------------------------------------------------------
# 進階版 Coordinate Attention (結合 Mean 與 Max 分支)
# --------------------------------------------------------
class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAttMeanMax, self).__init__()
        # 降維比例，避免參數量過大，最低保留 8 個 Channel
        mip = max(8, inp // reduction)

        # X 軸與 Y 軸的平均與最大池化
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        # 共享的 1x1 卷積層進行特徵壓縮
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        # 分別生成 X 軸與 Y 軸注意力權重的卷積層
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 1. 垂直方向 (H) 池化：Mean + Max
        x_h_avg = self.pool_h_avg(x)
        x_h_max = self.pool_h_max(x)
        x_h = x_h_avg + x_h_max  # Shape: (N, C, H, 1)
        
        # 2. 水平方向 (W) 池化：Mean + Max
        x_w_avg = self.pool_w_avg(x)
        x_w_max = self.pool_w_max(x)
        x_w = x_w_avg + x_w_max  # Shape: (N, C, 1, W)
        
        # 3. 空間維度拼接 (Concatenate) 並進行 1x1 卷積特徵轉換
        # 注意：需要把 W 方向的張量轉置才能和 H 方向拼接
        y = torch.cat([x_h, x_w.transpose(2, 3)], dim=2) # Shape: (N, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # 4. 將特徵切分回 H 與 W 方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)
        
        # 5. 透過 Sigmoid 函數產生最終的座標注意力權重 (0 ~ 1)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        # 6. 將權重乘回原始特徵
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
            self.backbone = models.resnet50(weights=weights)
            self.backbone.fc = nn.Identity()
            
            # 註冊攔截器
            self.backbone.layer1.register_forward_hook(self.get_hook('layer1'))
            self.backbone.layer2.register_forward_hook(self.get_hook('layer2'))
            self.backbone.layer3.register_forward_hook(self.get_hook('layer3'))
            self.backbone.layer4.register_forward_hook(self.get_hook('layer4'))
            
            # 🌟 將原本的 CBAM 替換為 CoordAttMeanMax
            self.coordatt4 = CoordAttMeanMax(2048)
            self.coordatt3 = CoordAttMeanMax(1024)
            self.coordatt2 = CoordAttMeanMax(512)
            self.coordatt1 = CoordAttMeanMax(256)
            
            # FPN 轉換
            self.fpn_latlayer4 = nn.Conv2d(2048, 256, kernel_size=1)
            self.fpn_latlayer3 = nn.Conv2d(1024, 256, kernel_size=1)
            self.fpn_latlayer2 = nn.Conv2d(512, 256, kernel_size=1)
            self.fpn_latlayer1 = nn.Conv2d(256, 256, kernel_size=1)
            
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            self.holographic_dim = 1024
            # 特徵分類頭
            self.classifier_head = nn.Linear(self.holographic_dim, num_classes)
            
            # 階層對比頭
            self.projection_head = nn.Sequential(
                nn.Linear(self.holographic_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128)
            )
        else :
            raise ValueError("Model ERROR")

    def get_hook(self, layer_name):
        def hook_fn(module, input, output):
            self.feature_map[layer_name] = output
        return hook_fn

    # FPN (Feature Pyramid Network)    
    def forward(self, x):      
        _ = self.backbone(x)
        
        # 🌟 透過 CoordAtt 過濾 Backbone 提取的特徵
        c4 = self.coordatt4(self.feature_map['layer4'])
        c3 = self.coordatt3(self.feature_map['layer3'])
        c2 = self.coordatt2(self.feature_map['layer2'])
        c1 = self.coordatt1(self.feature_map['layer1'])
        
        p4 = self.fpn_latlayer4(c4)
        p4_upsampled = F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        
        p3 = self.fpn_latlayer3(c3) + p4_upsampled
        p3_upsampled = F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        
        p2 = self.fpn_latlayer2(c2) + p3_upsampled
        p2_upsampled = F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        
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
        
        holographic_vector = torch.cat([pool_p4, pool_p3, pool_p2, pool_p1], dim=1)
        
        # 分類結果與對比空間投影
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