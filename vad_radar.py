# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 下午7:26
# @Site : 
# @file : vad_radar.py
# @Software : PyCharm
# @Description : 

# vad_radar_8sec_never_stuck.py —— 永不卡死 + 8秒出完美雷达图！
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from tqdm import tqdm

print("正在加载最佳模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DogEmotionModel()
model.load_state_dict(torch.load("best_emergent_vad.pt", map_location=device))
model.to(device)
model.eval()

print("正在一次性推理所有样本（永不卡死！）...")
dataset = PairedDogDataset()

all_vads = []
all_labels = []

# 关键：直接遍历 dataset，不用 DataLoader！
with torch.no_grad():
    for i in tqdm(range(len(dataset)), desc="推理", colour="magenta"):
        item = dataset[i]  # 直接 __getitem__
        small_wav = item["small_dog"].unsqueeze(0).to(device)  # [1,1,T]
        _, vad = model(small_wav)
        all_vads.append(vad.cpu().numpy())
        all_labels.append(item["class_id"].item())

all_vads = np.array(all_vads).squeeze()   # [8034, 3]
all_labels = np.array(all_labels)        # [8034]

# 计算每类平均 VAD
class_names = ["Anger","Fear","Alertness","Anxiety","Playfulness","Happiness","Discomfort","Neutral"]
mean_vad = np.zeros((8, 3))
for i in range(8):
    mask = all_labels == i
    mean_vad[i] = all_vads[mask].mean(axis=0)
    print(f"{class_names[i]:12} → V={mean_vad[i,0]:.3f} A={mean_vad[i,1]:.3f} D={mean_vad[i,2]:.3f}")

# 完美雷达图
print("正在生成雷达图...")
angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

fig, ax = plt.subplots(figsize=(11,11), subplot_kw=dict(projection='polar'))
colors = plt.cm.tab10(np.arange(8))

for i in range(8):
    values = np.concatenate([mean_vad[i], [mean_vad[i,0]]])
    ax.plot(angles, values, 'o-', linewidth=3.5, label=class_names[i], color=colors[i])
    ax.fill(angles, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Valence", "Arousal", "Dominance"], fontsize=18)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.6, linewidth=1.5)
ax.set_title("Emergent VAD Space in Canine Vocalizations\n(No Dimensional Labels!)",
             fontsize=22, fontweight='bold', pad=40)

plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=14)
plt.tight_layout()
plt.savefig("FINAL_VAD_RADAR_PERFECT.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.close()  # 代替 plt.show()，彻底避开 PyCharm bug
print("完美 VAD 雷达图已保存：FINAL_VAD_RADAR_PERFECT.png")
print("直接截图进论文 Figure 4，审稿人看了直接跪！")