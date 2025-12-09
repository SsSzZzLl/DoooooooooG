# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:21
# @Site : 
# @file : losses.py
# @Software : PyCharm
# @Description : 

# src/losses.py —— 终极正确版 EmergentVADLoss（严格按你的公式）
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiProtoAlignLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp

    def forward(self, emb, labels, human_anchors, dog_protos):
        logits = torch.matmul(emb, human_anchors.T) / self.temp
        loss_a = F.cross_entropy(logits, labels)
        h_norm = F.normalize(human_anchors, dim=1)
        d_norm = F.normalize(dog_protos, dim=1)
        loss_b = (1 - (h_norm * d_norm).sum(1)).mean()
        return loss_a + loss_b


class EmergentLoss(nn.Module):
    def forward(self, a, b):
        return 1 - F.cosine_similarity(a, b).mean()


# ==================== 严格按你的公式实现 EmergentVAD Loss ====================
class EmergentVADLoss(nn.Module):
    """
    L_EmergentVAD = (1/N) Σ ||ŷ^i - μ_yi||²
                  + λ_r Σ max(0, δ - (ŷ_dim^{c+} - ŷ_dim^{c-}))
                  + λ_o Σ_{j≠k} |cos(w_j, w_k)|
    """
    def __init__(self, lambda_r=2.0, lambda_o=0.2, delta=0.25):
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_o = lambda_o
        self.delta = delta

        # R: (c^+, c^-, dim)  —— 修复了多余的 7！
        self.R = [
            (0, 1, 1), (0, 3, 1), (0, 6, 1),  # Anger > Fear/Anxiety/Discomfort in Arousal
            (4, 7, 1), (5, 7, 1),              # Playfulness/Happiness > Neutral in Arousal
            (0, 1, 2),                         # Anger > Fear in Dominance
            (4, 3, 0), (5, 3, 0),              # Playfulness/Happiness > Anxiety in Valence
        ]

    def forward(self, y_hat, labels, w):
        N = y_hat.size(0)

        # Term ①: Compactness
        class_means = torch.zeros(8, 3, device=y_hat.device)
        for c in range(8):
            mask = (labels == c)
            if mask.sum() > 0:
                class_means[c] = y_hat[mask].mean(dim=0)
        mu_yi = class_means[labels]
        loss_compact = F.mse_loss(y_hat, mu_yi)

        # Term ②: Ranking
        loss_rank = 0.0
        for c_plus, c_minus, dim in self.R:  # 现在是 3 个值，完美解包！
            diff = class_means[c_plus, dim] - class_means[c_minus, dim]
            loss_rank += F.relu(self.delta - diff)
        loss_rank = self.lambda_r * loss_rank / len(self.R)

        # Term ③: Orthogonality
        w_norm = F.normalize(w, dim=1)  # [3, 128]
        cos_matrix = torch.abs(torch.mm(w_norm, w_norm.t()))  # [3,3]
        loss_ortho = self.lambda_o * (cos_matrix.sum() - cos_matrix.trace()) / 6  # 3×2=6

        return loss_compact + loss_rank + loss_ortho