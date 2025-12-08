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