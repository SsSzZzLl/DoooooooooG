# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 下午7:26
# @Site : 
# @file : vad_radar.py
# @Software : PyCharm
# @Description : 

# vad_radar.py —— 永不卡死 + 自动时间戳保存
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from tqdm import tqdm
from datetime import datetime

# 自动时间戳保存路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"runs/run_{timestamp.split('_')[0]}"  # 和训练同目录
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"vad_radar_{timestamp}.png")

print("正在加载最佳模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DogEmotionModel()
model.load_state_dict(torch.load("runs/run_*/best_model.pt", map_location=device))  # 改成你的路径
model.to(device)
model.eval()

print("正在推理（永不卡死版本）...")
dataset = PairedDogDataset()
all_vads, all_labels = [], []

with torch.no_grad():
    for i in tqdm(range(len(dataset)), desc="推理", colour="magenta"):
        item = dataset[i]
        wav = item["small_dog"].unsqueeze(0).to(device)
        _, vad = model(wav)
        all_vads.append(vad.cpu().numpy())
        all_labels.append(item["class_id"].item())

all_vads = np.array(all_vads).squeeze()
all_labels = np.array(all_labels)

# 计算平均 VAD
names = ["Anger","Fear","Alertness","Anxiety","Playfulness","Happiness","Discomfort","Neutral"]
mean_vad = np.zeros((8,3))
for i in range(8):
    mean_vad[i] = all_vads[all_labels == i].mean(0)
    print(f"{names[i]:12} → V={mean_vad[i,0]:.3f} A={mean_vad[i,1]:.3f} D={mean_vad[i,2]:.3f}")

# 完美雷达图
angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

fig, ax = plt.subplots(figsize=(11,11), subplot_kw=dict(projection='polar'))
colors = plt.cm.tab10(np.arange(8))

for i in range(8):
    values = np.concatenate([mean_vad[i], [mean_vad[i,0]]])
    ax.plot(angles, values, 'o-', linewidth=4, label=names[i], color=colors[i])
    ax.fill(angles, values, alpha=0.2, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Valence", "Arousal", "Dominance"], fontsize=18)
ax.set_ylim(0,1)
ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
ax.grid(True, linewidth=1.5)
plt.title("Emergent VAD Space in Canine Vocalizations\n(Zero-Shot Cross-Species)",
          fontsize=24, fontweight='bold', pad=40)
plt.legend(loc='upper right', bbox_to_anchor=(1.35,1.0), fontsize=14)
plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n完美雷达图已保存：{save_path}")
print("直接截图进论文 Figure 4，一作稳了！")