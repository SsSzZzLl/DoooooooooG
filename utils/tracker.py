# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:21
# @Site : 
# @file : tracker.py
# @Software : PyCharm
# @Description : 

# utils/tracker.py
import torch

class PrototypeTracker:
    def __init__(self, num_classes=8, dim=128, momentum=0.9, device='cpu'):
        self.protos = torch.zeros(num_classes, dim).to(device)
        self.momentum = momentum
        self.device = device
        self.init = False

    def update(self, emb, labels):
        with torch.no_grad():
            for c in range(8):
                mask = (labels == c)
                if mask.sum() > 0:
                    mean = emb[mask].mean(0)
                    if not self.init:
                        self.protos[c] = mean
                    else:
                        self.protos[c] = self.momentum * self.protos[c] + (1 - self.momentum) * mean
            self.init = True

    def get_prototypes(self):
        return self.protos.detach()