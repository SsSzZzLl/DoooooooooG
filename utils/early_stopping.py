# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午11:00
# @Site : 
# @file : early_stopping.py
# @Software : PyCharm
# @Description : 

# utils/early_stopping.py
import torch

class EarlyStopping:
    def __init__(self, patience=5, save_path="best_model.pt"):
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best = float('inf')

    def __call__(self, val_loss, model):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                return True
            return False