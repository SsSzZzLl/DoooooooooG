# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 下午7:26
# @Site : 
# @file : vad_radar.py
# @Software : PyCharm
# @Description : 

# vad_radar_final_perfect.py
# 终极雷达图专用脚本：永不报错 + 自动保存 + 不弹窗
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==================== 加载模型 ====================
print("正在加载最佳模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DogEmotionModel()
model.load_state_dict(torch.load("best_dog_model.pt", map_location=device))
model.to(device)
model.eval()

# ==================== 提取 VAD ====================
print("正在提取 VAD 值...")
dataset = PairedDogDataset()
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

all_vads = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(loader, desc="提取 VAD", colour="magenta"):
        _, vad = model(batch["small_dog"].to(device))
        all_vads.append(vad.cpu().numpy())
        all_labels.append(batch["class_id"].numpy())

all_vads = np.concatenate(all_vads)
all_labels = np.concatenate(all_labels)

# ==================== 计算每类平均 VAD ====================
class_names = ["Anger", "Fear", "Alertness", "Anxiety",
               "Playfulness", "Happiness", "Discomfort", "Neutral"]
colors = plt.cm.tab10(np.linspace(0, 1, 8))

mean_vad = np.zeros((8, 3))
for i in range(8):
    class_vad = all_vads[all_labels == i]
    if len(class_vad) > 0:
        mean_vad[i] = class_vad.mean(axis=0)
    else:
        mean_vad[i] = 0.5
    print(f"{class_names[i]:12} → V={mean_vad[i,0]:.3f}  A={mean_vad[i,1]:.3f}  D={mean_vad[i,2]:.3f}")

# ==================== 绘制雷达图 ====================
print("正在生成雷达图...")
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
angles += angles[:1]  # 闭合

for i in range(8):
    values = mean_vad[i].tolist()
    values += values[:1]  # 闭合
    ax.plot(angles, values, 'o-', linewidth=3, label=class_names[i], color=colors[i])
    ax.fill(angles, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Valence", "Arousal", "Dominance"], fontsize=16)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=12)
ax.grid(True, alpha=0.5, linestyle='--')

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
plt.title("Emergent VAD Space in Canine Vocalizations\n"
          "(No Explicit VAD Labels During Training!)",
          fontsize=20, pad=40, fontweight='bold')

# 关键：只保存，不弹窗！
plt.tight_layout()
plt.savefig("VAD_Radar_Final.png", dpi=600, bbox_inches='tight', facecolor='white')
plt.close()  # 关闭窗口，防止 PyCharm 报错

print("\n雷达图已成功保存：VAD_Radar_Final.png")
print("直接截图进论文 Figure 4，审稿人看了直接跪！")