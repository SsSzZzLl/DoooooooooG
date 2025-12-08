# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:45
# @Site : 
# @file : early_stopping.py
# @Software : PyCharm
# @Description : 

# utils/early_stopping.py
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, save_path="best.pt"):
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best = float('inf')

    def __call__(self, val_loss, model):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False