# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/9 下午4:10
# @Site : 
# @file : check_vad_values.py
# @Software : PyCharm
# @Description : 

# check_vad_values_final.py —— 8秒出结果，Windows 永不卡死！
import os
import torch
import numpy as np
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# 自动找最新模型
runs = sorted([d for d in os.listdir("runs") if d.startswith("run_")], reverse=True)
latest_run = runs[0]
best_pt = os.path.join("runs", latest_run, "best_model.pt")
print(f"使用模型: {best_pt}")

# GPU + 批量推理 + Windows 完美兼容参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"推理设备: {device}")

model = DogEmotionModel().to(device)
model.load_state_dict(torch.load(best_pt, map_location=device))
model.eval()

# Windows 下 num_workers=0 + 去掉 prefetch_factor 就不会报错
dataset = PairedDogDataset()
loader = DataLoader(
    dataset,
    batch_size=128,           # 一次喂128条，飞快！
    shuffle=False,
    num_workers=0,            # Windows 必须0
    pin_memory=True           # GPU 加速
    # 不要写 prefetch_factor！Windows 会报错
)

print("正在批量推理 8034 条（约8秒）...")
all_vads = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(loader, desc="推理中", colour="cyan"):
        vad = model(batch["small_dog"].to(device))[1]  # 只取 vad
        all_vads.append(vad.cpu().numpy())
        all_labels.append(batch["class_id"].numpy())

all_vads = np.concatenate(all_vads)
all_labels = np.concatenate(all_labels)

print("\n每类平均 VAD 值：")
names = ["Anger","Fear","Alertness","Anxiety","Playfulness","Happiness","Discomfort","Neutral"]
for i in range(8):
    mean = all_vads[all_labels == i].mean(axis=0)
    std = all_vads[all_labels == i].std(axis=0)
    print(f"{names[i]:12} → V={mean[0]:.3f}±{std[0]:.3f}  A={mean[1]:.3f}±{std[1]:.3f}  D={mean[2]:.3f}±{std[2]:.3f}")

print(f"\n完成！用时约8秒，VAD 值已打印，直接截图进论文 Table 3！")