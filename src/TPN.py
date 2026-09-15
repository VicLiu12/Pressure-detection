import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import linprog

class Cost_Focal_Loss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, l2_reg=0.1):
        super(Cost_Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l2_reg = l2_reg

        prior_matrix = [
            [1.0, 5.0, 2.0, 3.0, 4.0, 5.0, 5.0], 
            [6.0, 1.0, 4.0, 3.0, 3.0, 4.0, 3.0], 
            [2.0, 3.0, 1.0, 1.5, 3.0, 5.0, 4.0], 
            [4.0, 2.0, 5.0, 1.0, 1.5, 3.0, 3.0], 
            [6.0, 3.0, 4.0, 1.5, 1.0, 1.5, 1.5], 
            [8.0, 4.0, 6.0, 3.0, 1.5, 1.0, 1.5], 
            [8.0, 3.0, 6.0, 4.0, 1.5, 1.5, 1.0]  
        ]
        self.register_buffer("prior_matrix", torch.tensor(prior_matrix, dtype=torch.float32))
        self.dynamic_matrix = nn.Parameter(torch.tensor(prior_matrix, dtype=torch.float32))
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        probs = F.softmax(inputs, dim=1)
        targets_costs = self.dynamic_matrix[targets]
        expected_costs = torch.sum(probs * targets_costs, dim=1)
        weighted_loss = focal_loss * expected_costs
        final_loss = weighted_loss.mean()
        reg_loss = self.l2_reg * torch.norm(self.dynamic_matrix - self.prior_matrix)
        return final_loss + reg_loss
    
class OrdinalSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(OrdinalSupConLoss, self).__init__()
        self.temperature = temperature
        ordinal_levels = [0.0, 3.5, 1.0, 2.0, 3.0, 4.0, 3.5]
        self.register_buffer("ordinal_levels", torch.tensor(ordinal_levels, dtype=torch.float32))
        
    def forward(self, feature, labels):
        device = feature.device
        batch_size = feature.shape[0]
        sim_matrix = torch.matmul(feature, feature.T) / self.temperature
        sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_matrix_max.detach()
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        sample_levels = self.ordinal_levels[labels.squeeze()]
        level_diff = torch.abs(sample_levels.view(-1, 1) - sample_levels.view(1, -1))
        distance_weight = torch.ones_like(level_diff) + (1.0 - mask) * level_diff
        exp_logits = torch.exp(logits) * logits_mask * distance_weight
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        return loss

# --------------------------------------------------------
# Wasserstein 單峰投影演算法 (線性規劃)
# --------------------------------------------------------
def unimodal_wasserstein(p, mode):
    # 確保機率總和為 1，避免浮點數誤差導致線性規劃無解
    p = p / np.sum(p)
    K = p.size
    C = squareform(pdist(np.arange(K)[:, None]))
    Ap = [([0]*i + [1] + [0]*(K-i-1))*K for i in range(K)]
    Ai = [[0]*i*K + [1]*K + [-1]*K + [0]*(K-i-2)*K if i < mode else
          [0]*i*K + [-1]*K + [1]*K + [0]*(K-i-2)*K for i in range(K-1)]
    
    # 使用 highs 演算法加速求解
    result = linprog(C.ravel(), A_ub=Ai, b_ub=np.zeros(K-1), A_eq=Ap, b_eq=p, bounds=(0, None), method='highs')
    T = result.x.reshape(K, K)
    return (T*C).sum(), T.sum(1)

class WassersteinRegularizationLoss(nn.Module):
    def __init__(self):
        super(WassersteinRegularizationLoss, self).__init__()

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs_log = F.log_softmax(logits, dim=1)
        device = logits.device
        
        target_unimodal = []
        # 將 Tensor 轉至 CPU 進行 scipy 線性規劃運算 (效能瓶頸所在)
        for phat, y in zip(probs.cpu().detach().numpy(), targets.cpu().numpy()):
            _, closest_dist = unimodal_wasserstein(phat, y)
            target_unimodal.append(torch.tensor(closest_dist, dtype=torch.float32, device=device))
        
        target_unimodal = torch.stack(target_unimodal)
        
        # 使用 KL 散度計算當前分佈與線性規劃求得的最理想單峰分佈差距
        uni_loss = torch.sum(F.kl_div(probs_log, target_unimodal, reduction='none'), dim=1).mean()
        return uni_loss

class JoinLoss(nn.Module):
    def __init__(self, alpha=0, gamma=2.0, l2_reg=0.1, lambda_con=0.5, temperature=0.07, lambda_uni=1.0):
        super(JoinLoss, self).__init__()
        self.cls_closs_fn = Cost_Focal_Loss(alpha=alpha, gamma=gamma, l2_reg=l2_reg)
        self.con_loss_fn = OrdinalSupConLoss(temperature=temperature)
        self.uni_loss_fn = WassersteinRegularizationLoss()

        self.lambda_con = lambda_con
        self.lambda_uni = lambda_uni
        
    def forward(self, classification_result, projected_feature, targets):
        cls_loss = self.cls_closs_fn(classification_result, targets)
        con_loss = self.con_loss_fn(projected_feature, targets)
        uni_loss = self.uni_loss_fn(classification_result, targets)
        
        total_loss = cls_loss + (self.lambda_con * con_loss) + (self.lambda_uni * uni_loss)
        return total_loss, cls_loss, con_loss, uni_loss