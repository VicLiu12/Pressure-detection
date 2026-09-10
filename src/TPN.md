import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# 1. 保留原本的 Cost_Focal_Loss 與 OrdinalSupConLoss
# --------------------------------------------------------
class Cost_Focal_Loss(nn.Module):
    def __init__(self, alpha = 1.0, gamma = 2.0, l2_reg = 0.1):
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
        self.register_buffer("prior_matrix", torch.tensor(prior_matrix, dtype = torch.float32))
        self.dynamic_matrix = nn.Parameter(torch.tensor(prior_matrix, dtype = torch.float32))
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction = 'none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        probs = F.softmax(inputs, dim = 1)
        targets_costs = self.dynamic_matrix[targets]
        expected_costs = torch.sum(probs * targets_costs, dim = 1)
        weighted_loss = focal_loss * expected_costs
        final_loss = weighted_loss.mean()
        reg_loss = self.l2_reg * torch.norm(self.dynamic_matrix - self.prior_matrix)
        return final_loss + reg_loss
    
class OrdinalSupConLoss(nn.Module):
    def __init__(self, temperature = 0.07):
        super(OrdinalSupConLoss, self).__init__()
        self.temperature = temperature
        ordinal_levels = [0.0, 3.5, 1.0, 2.0, 3.0, 4.0, 3.5]
        self.register_buffer("ordinal_levels", torch.tensor(ordinal_levels, dtype = torch.float32))
        
    def forward(self, feature, labels):
        device = feature.device
        batch_size = feature.shape[0]
        sim_matrix = torch.matmul(feature, feature.T) / self.temperature
        sim_matrix_max, _ = torch.max(sim_matrix, dim = 1, keepdim = True)
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
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim = True) + 1e-12)
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        return loss

# --------------------------------------------------------
# 2. 新增：全 GPU 加速單峰正則化損失 (Soft Unimodal Loss)
# --------------------------------------------------------
class UnimodalRegularizationLoss(nn.Module):
    def __init__(self):
        super(UnimodalRegularizationLoss, self).__init__()

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        batch_size, num_classes = probs.shape
        device = probs.device

        # 計算相鄰類別機率的差值 (p_{i+1} - p_i)
        diff = probs[:, 1:] - probs[:, :-1]  # Shape: (Batch, num_classes-1)
        
        idx = torch.arange(num_classes, device=device).unsqueeze(0).expand(batch_size, -1)
        true_class = targets.unsqueeze(1)

        # 真實類別左側：機率應該要遞增 (diff >= 0)，若 diff < 0 則給予懲罰 (-diff)
        left_mask = (idx[:, 1:] <= true_class).float()
        left_penalty = F.relu(-diff) * left_mask

        # 真實類別右側：機率應該要遞減 (diff <= 0)，若 diff > 0 則給予懲罰 (diff)
        right_mask = (idx[:, :-1] >= true_class).float()
        right_penalty = F.relu(diff) * right_mask

        # 加總左右兩側的非單峰懲罰值
        loss = (left_penalty + right_penalty).sum(dim=1).mean()
        return loss

# --------------------------------------------------------
# 3. 升級 JoinLoss (結合三股力量)
# --------------------------------------------------------
class JoinLoss(nn.Module):
    def __init__(self, alpha=0, gamma=2.0, l2_reg=0.1, lambda_con=0.5, temperature=0.07, lambda_uni=1.0):
        super(JoinLoss, self).__init__()
        self.cls_closs_fn = Cost_Focal_Loss(alpha=alpha, gamma=gamma, l2_reg=l2_reg)
        self.con_loss_fn = OrdinalSupConLoss(temperature=temperature)
        self.uni_loss_fn = UnimodalRegularizationLoss() # 引入單峰正則化
        
        self.lambda_con = lambda_con
        self.lambda_uni = lambda_uni
        
    def forward(self, classification_result, projected_feature, targets):
        cls_loss = self.cls_closs_fn(classification_result, targets)
        con_loss = self.con_loss_fn(projected_feature, targets)
        uni_loss = self.uni_loss_fn(classification_result, targets)
        
        total_loss = cls_loss + (self.lambda_con * con_loss) + (self.lambda_uni * uni_loss)
        
        # 🌟 回傳 4 個變數
        return total_loss, cls_loss, con_loss, uni_loss

# --------------------------------------------------------
# 測試區塊
# --------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_classes = 7
    latent_dim = 128
        
    mock_class_out = torch.randn(batch_size, num_classes, requires_grad=True).to(device)
    mock_proj_out = F.normalize(torch.randn(batch_size, latent_dim, requires_grad=True), p=2, dim=1).to(device)
    mock_targets = torch.randint(0, num_classes, (batch_size,)).to(device)
        
    criterion = JoinLoss(lambda_con=0.5, lambda_uni=1.0).to(device)
    total_loss, cls_loss, con_loss, uni_loss = criterion(mock_class_out, mock_proj_out, mock_targets)
    
    print(f"Total loss : {total_loss.item():.4f}")
    print(f"- Cls loss : {cls_loss.item():.4f}")
    print(f"- Con loss : {con_loss.item():.4f}")
    print(f"- Uni loss : {uni_loss.item():.4f}")








# 原本：loss, cls_loss, con_loss = criterion(class_out, proj_out, labels)
loss, cls_loss, con_loss, uni_loss = criterion(class_out, proj_out, labels)

# 原本：loss_2, _, _ = criterion(class_out_2, proj_out_2, labels)
loss_2, _, _, _ = criterion(class_out_2, proj_out_2, labels)