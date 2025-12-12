# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:53
# @Site : 
# @file : checkpoint.py
# @Software : PyCharm
# @Description : 

# utils/checkpoint.py
import torch
import os

def save_checkpoint(exp_dir, epoch, model, optimizer, dynamic_anchors, tracker, history, best_val):
    path = os.path.join(exp_dir, "checkpoint_latest.pt")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "dynamic_anchors": dynamic_anchors.cpu(),
        "tracker": tracker.state_dict() if hasattr(tracker, 'state_dict') else tracker,
        "history": history,
        "best_val": best_val,
    }, path)
    print(f"检查点已保存: {path}")

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint