import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_function(nn.Module):
    
    def __init__(self, alpha = 1.0, gamma = 2.0, l2_reg = 0.1):
        super(Loss_function, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l2_reg = l2_reg
        
        #[Invalid, SDTI, Stage_I, Stage_II, Stage_III, Stage_IV, Unstageable]
        prior_matrix = [
            [1.0, 5.0, 2.0, 3.0, 4.0, 5.0, 5.0], #Invalid
            [6.0, 1.0, 4.0, 3.0, 3.0, 4.0, 3.0], #SDTI
            [2.0, 3.0, 1.0, 1.5, 3.0, 5.0, 4.0], #Stage_I
            [4.0, 2.0, 1.5, 1.0, 1.5, 3.0, 3.0], #Stage_II
            [6.0, 3.0, 4.0, 1.5, 1.0, 1.5, 1.5], #Stage_III
            [8.0, 4.0, 6.0, 3.0, 1.5, 1.0, 1.5], #Stage_IV
            [8.0, 3.0, 6.0, 4.0, 1.5, 1.5, 1.0]  #Unstageble
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


