# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午11:00
# @Site : 
# @file : tracker.py
# @Software : PyCharm
# @Description : 

# utils/tracker.py
import torch
import torch.nn.functional as F

class PrototypeTracker:
    def __init__(self, num_classes=8, dim=128, device="cpu"):
        self.num_classes = num_classes
        self.dim = dim
        self.device = device
        self.prototypes = torch.zeros(num_classes, dim, device=device)
        self.counts = torch.zeros(num_classes, device=device)

    def update(self, embeddings, labels):
        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                class_emb = embeddings[mask]
                self.prototypes[i] = 0.99 * self.prototypes[i] + 0.01 * class_emb.mean(0)
                self.counts[i] += mask.sum()

    def get_prototypes(self):
        valid = self.counts > 0
        protos = self.prototypes.clone()
        protos[~valid] = 0
        return F.normalize(protos, dim=1)

    def state_dict(self):
        return {"prototypes": self.prototypes, "counts": self.counts}

    def load_state_dict(self, state):
        self.prototypes = state["prototypes"].to(self.device)
        self.counts = state["counts"].to(self.device)