# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:21
# @Site : 
# @file : losses.py
# @Software : PyCharm
# @Description : 

# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiProtoAlignLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, emb, labels, human_anchors, dog_protos, temp=0.07):
        logits = torch.matmul(emb, human_anchors.T) / temp
        loss_a = F.cross_entropy(logits, labels)
        h_norm = F.normalize(human_anchors, dim=1)
        d_norm = F.normalize(dog_protos, dim=1)
        loss_b = (1 - (h_norm * d_norm).sum(1)).mean()
        return loss_a + loss_b

class EmergentLoss(nn.Module):
    def forward(self, a, b):
        return 1 - F.cosine_similarity(a, b).mean()

class EmergentVADLoss(nn.Module):
    def __init__(self, lambda_r=2.0, lambda_o=0.2, delta=0.25):
        super().__init__()
        self.lambda_r = lambda_r
        self.lambda_o = lambda_o
        self.delta = delta

        # (c+, c-, dim, weight) —— Dominance 规则权重×3倍！
        self.rules = [
            (0, 1, 1, 1.0), (0, 3, 1, 1.0), (0, 6, 1, 1.0),
            (4, 7, 1, 1.0), (5, 7, 1, 1.0),
            (0, 1, 2, 3.0),  # Anger > Fear 在 Dominance 上最重要！
            (4, 3, 0, 1.0), (5, 3, 0, 1.0),
        ]

    def forward(self, y_hat, labels, w):
        N = y_hat.size(0)

        # ① Compactness（余弦版）
        class_means = torch.zeros(8, 3, device=y_hat.device)
        for c in range(0, 8):
            mask = (labels == c)
            if mask.sum() > 0:
                class_means[c] = y_hat[mask].mean(0)
        mu_yi = class_means[labels]
        loss_compact = 1 - F.cosine_similarity(y_hat, mu_yi).mean()

        # ② Ranking（加权）
        loss_rank = 0.0
        total_weight = 0.0
        for c_plus, c_minus, dim, weight in self.rules:
            diff = class_means[c_plus, dim] - class_means[c_minus, dim]
            loss_rank += weight * F.relu(self.delta - diff)
            total_weight += weight
        loss_rank = self.lambda_r * loss_rank / total_weight

        # ③ Orthogonality
        w_norm = F.normalize(w, dim=1)
        cos_matrix = torch.abs(torch.mm(w_norm, w_norm.t()))
        loss_ortho = self.lambda_o * (cos_matrix.sum() - cos_matrix.trace()) / 6

        return loss_compact + loss_rank + loss_ortho