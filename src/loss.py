import torch
import torch.nn as nn
import torch.nn.functional as F

class Cost_Focal_Loss(nn.Module):
    
    def __init__(self, alpha = 1.0, gamma = 2.0, l2_reg = 0.1):
        super(Cost_Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l2_reg = l2_reg


        #Cost Sensitive Learning & Prior Matrix
        #[Invalid, SDTI, Stage_I, Stage_II, Stage_III, Stage_IV, Unstageable]
        prior_matrix = [
            [1.0, 5.0, 2.0, 3.0, 4.0, 5.0, 5.0], #Invalid
            [6.0, 1.0, 4.0, 3.0, 3.0, 4.0, 3.0], #SDTI
            [2.0, 3.0, 1.0, 1.5, 3.0, 5.0, 4.0], #Stage_I
            [4.0, 2.0, 5.0, 1.0, 1.5, 3.0, 3.0], #Stage_II
            [6.0, 3.0, 4.0, 1.5, 1.0, 1.5, 1.5], #Stage_III
            [8.0, 4.0, 6.0, 3.0, 1.5, 1.0, 1.5], #Stage_IV
            [8.0, 3.0, 6.0, 4.0, 1.5, 1.5, 1.0]  #Unstageble
        ]

        #將Prior Matrix作為固定參考，不會餐與梯度更新
        self.register_buffer("prior_matrix", torch.tensor(prior_matrix, dtype = torch.float32))

        #將同樣的矩陣設為dynamic matrix，代表這個矩陣可以隨著神經網路一起被訓練以及微調
        self.dynamic_matrix = nn.Parameter(torch.tensor(prior_matrix, dtype = torch.float32))
        

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction = 'none')
        #Focal Loss焦點損失函數
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        probs = F.softmax(inputs, dim = 1)
        
        targets_costs = self.dynamic_matrix[targets]

        #Expected Costs期望代價 --> 期望值的概念
        expected_costs = torch.sum(probs * targets_costs, dim = 1)
        
        weighted_loss = focal_loss * expected_costs
        final_loss = weighted_loss.mean()

        #L2正規畫 (L2 Regularization / Weight Decy) --> 避免dynamic matrix出現極端狀況
        reg_loss = self.l2_reg * torch.norm(self.dynamic_matrix - self.prior_matrix)
        
        return final_loss + reg_loss
    
class OrdinalSupConLoss(nn.Module):
    def __init__(self, temperature = 0.07):
        super(OrdinalSupConLoss, self).__init__()
        self.temperature = temperature
        
        #不同分類的相互推開的力度
        #[Ivalid, SDTI, StageI, StageII, StageIII, StageIV, Unstageable]
        ordinal_levels = [0.0, 3.5, 1.0, 2.0, 3.0, 4.0, 3.5]
        self.register_buffer("ordinal_levels", torch.tensor(ordinal_levels, dtype = torch.float32))
        
    def forward(self, feature, labels):
        #feature : 來自 projection_head 且已 L2 Normalize 的特徵
        device = feature.device
        batch_size = feature.shape[0]
        
        #計算Batch中樣本間的相似度距離
        sim_matrix = torch.matmul(feature, feature.T) / self.temperature
        
        sim_matrix_max, _ = torch.max(sim_matrix, dim = 1, keepdim = True)
        logits = sim_matrix - sim_matrix_max.detach()
        
        #同類別的樣本位置設為 1
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        #計畫Batch內的階層距離，距離越遠則給予越大的排斥利
        sample_levels = self.ordinal_levels[labels.squeeze()]
        level_diff = torch.abs(sample_levels.view(-1, 1) - sample_levels.view(1, -1))
        distance_weight = torch.ones_like(level_diff) + (1.0 - mask) * level_diff
        
        


