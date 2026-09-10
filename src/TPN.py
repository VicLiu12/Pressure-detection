import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import linprog

# --------------------------------------------------------
# 1. 移植文獻中的 Wasserstein 單峰投影演算法
# --------------------------------------------------------
def unimodal_wasserstein(p, mode):
    """
    計算距離機率分佈 p 最近的單峰分佈 (以 mode 為峰值)
    回傳: (總傳輸代價, 最近的單峰分佈)
    """
    assert abs(p.sum()-1) < 1e-6, 'Expected normalized probability mass.'
    assert np.any(p >= 0), 'Expected nonnegative probabilities.'
    
    K = p.size
    C = squareform(pdist(np.arange(K)[:, None]))  # 建立成本矩陣
    Ap = [([0]*i + [1] + [0]*(K-i-1))*K for i in range(K)]
    Ai = [[0]*i*K + [1]*K + [-1]*K + [0]*(K-i-2)*K if i < mode else
          [0]*i*K + [-1]*K + [1]*K + [0]*(K-i-2)*K for i in range(K-1)]
    
    result = linprog(C.ravel(), A_ub=Ai, b_ub=np.zeros(K-1), A_eq=Ap, b_eq=p)
    T = result.x.reshape(K, K)
    return (T*C).sum(), T.sum(1)

# --------------------------------------------------------
# (保留你原本的 Cost_Focal_Loss 與 OrdinalSupConLoss 類別)
# --------------------------------------------------------

# --------------------------------------------------------
# 2. 升級 JoinLoss，加入 Wasserstein 單峰正則化
# --------------------------------------------------------
class JoinLoss(nn.Module):
    def __init__(self, alpha=0, gamma=2.0, l2_reg=0.1, lambda_con=0.5, temperature=0.07, lambda_uni=1.0):
        super(JoinLoss, self).__init__()
        self.cls_closs_fn = Cost_Focal_Loss(alpha=alpha, gamma=gamma, l2_reg=l2_reg)
        self.con_loss_fn = OrdinalSupConLoss(temperature=temperature)
        
        self.lambda_con = lambda_con
        self.lambda_uni = lambda_uni  # 新增: 控制單峰正則化的權重 (建議初始設為 1.0 或 0.1)
        
    def forward(self, classification_result, projected_feature, targets):
        # 1. 計算分類誤判損失 (結合臨床代價矩陣)
        cls_loss = self.cls_closs_fn(classification_result, targets)
        
        # 2. 計算序數對比損失 (潛在空間分群)
        con_loss = self.con_loss_fn(projected_feature, targets)
        
        # 3. 計算 Wasserstein 單峰正則化損失
        probs = F.softmax(classification_result, dim=1)
        probs_log = F.log_softmax(classification_result, dim=1)
        device = classification_result.device
        
        target_unimodal = []
        # 將 Tensor 轉至 CPU 進行 scipy 線性規劃運算
        for phat, y in zip(probs.cpu().detach().numpy(), targets.cpu().numpy()):
            _, closest_dist = unimodal_wasserstein(phat, y)
            target_unimodal.append(torch.tensor(closest_dist, dtype=torch.float32, device=device))
        
        target_unimodal = torch.stack(target_unimodal)
        
        # 使用 KL 散度計算當前分佈與最理想單峰分佈的差距
        uni_loss = torch.sum(F.kl_div(probs_log, target_unimodal, reduction='none'), 1).mean()
        
        # 4. 總和所有損失
        total_loss = cls_loss + (self.lambda_con * con_loss) + (self.lambda_uni * uni_loss)
        
        # 回傳 total_loss 以及各項子損失以便在 train.py 中印出監控
        return total_loss, cls_loss, con_loss, uni_loss

# --------------------------------------------------------
# (測試區塊)
# --------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
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







