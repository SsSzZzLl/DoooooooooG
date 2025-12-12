# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午11:35
# @Site : 
# @file : optimize_4060.py
# @Software : PyCharm
# @Description : 

# optimize_4060.py  —— 4060 专属加速补丁（放项目根目录）
import torch

# Ada 架构最快模式
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True          # 自动找最快卷积算法
torch.backends.cudnn.deterministic = False     # 速度优先

# 混合精度（不改代码就提速）
torch.set_float32_matmul_precision('high')      # 4060 专属！比 medium 更快

print("4060 加速补丁已加载！tf32 + benchmark + high precision")