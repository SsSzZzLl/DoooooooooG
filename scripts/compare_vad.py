# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午11:22
# @Site : 
# @file : compare_vad.py
# @Software : PyCharm
# @Description : 

# scripts/compare_vad.py  —— 终极版：永不报错 + 自动创建文件夹 + 4060 优化
import sys
import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from datetime import datetime

# ==================== 自动定位项目根目录 ====================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# ==================== 读取配置 ====================
CONFIG_PATH = os.path.join(ROOT, "config", "experiments.yaml")
print(f"正在读取配置文件: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    configs = yaml.safe_load(f)
    cfg = configs["global"]
    infer_bs = cfg.get('infer_batch_size', 512)  # 默认512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device} | 推理 batch_size: {infer_bs}")

# ==================== 自动创建 experiments 文件夹 + 找最新 full 模型 ====================
EXP_ROOT = os.path.join(ROOT, "experiments")
os.makedirs(EXP_ROOT, exist_ok=True)  # 自动创建！

exp_dirs = [d for d in os.listdir(EXP_ROOT) if d.endswith("_full")]
if not exp_dirs:
    raise FileNotFoundError(f"未找到 full 模型！请先运行训练。检查目录: {EXP_ROOT}")

latest_exp = sorted(exp_dirs)[-1]
model_path = os.path.join(EXP_ROOT, latest_exp, "best_model.pt")
print(f"加载模型: {model_path}")

# ==================== 加载模型 + 推理狗的 VAD ====================
model = DogEmotionModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dataset = PairedDogDataset()
loader = DataLoader(
    dataset,
    batch_size=infer_bs,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

all_vad = []
all_labels = []
print("开始推理狗的 VAD...")
with torch.no_grad():
    for batch in loader:
        _, vad = model(batch["small_dog"].to(device))
        all_vad.append(vad.cpu())
        all_labels.append(batch["class_id"])

dog_vad = torch.cat(all_vad).numpy()
labels = torch.cat(all_labels).numpy()
mean_dog_vad = np.array([dog_vad[labels == i].mean(axis=0) for i in range(8)])

# ==================== 解码人类 VAD ====================
human_anchors = torch.load("real_human_anchors.pt", map_location=device)  # [8, 128]
with torch.no_grad():
    human_vad = model.vad_head(human_anchors).cpu().numpy()  # [8, 3]

# ==================== 生成表格 + 图 ====================
emotions = ["Anger", "Fear", "Alertness", "Anxiety", "Playfulness", "Happiness", "Discomfort", "Neutral"]

df = pd.DataFrame({
    "Emotion": emotions,
    "Human_V": np.round(human_vad[:,0], 4),
    "Human_A": np.round(human_vad[:,1], 4),
    "Human_D": np.round(human_vad[:,2], 4),
    "Dog_V": np.round(mean_dog_vad[:,0], 4),
    "Dog_A": np.round(mean_dog_vad[:,1], 4),
    "Dog_D": np.round(mean_dog_vad[:,2], 4),
})

# 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(ROOT, "experiments", "comparison", f"{timestamp}_vad_compare")
os.makedirs(out_dir, exist_ok=True)

df.to_excel(os.path.join(out_dir, "VAD_Human_vs_Dog.xlsx"), index=False)
df.to_csv(os.path.join(out_dir, "VAD_Human_vs_Dog.csv"), index=False)

# 画图（简单版）
plt.figure(figsize=(12, 8))
x = np.arange(len(emotions))
width = 0.35
plt.bar(x - width/2, human_vad.mean(axis=1), width, label="Human", alpha=0.8)
plt.bar(x + width/2, mean_dog_vad.mean(axis=1), width, label="Dog", alpha=0.8)
plt.xticks(x, emotions, rotation=45)
plt.ylabel("Average VAD Score")
plt.title("Human vs Dog VAD Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "VAD_Comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# 更新 latest 软链接（Windows 兼容）
latest_link = os.path.join(ROOT, "experiments", "comparison", "latest")
if os.path.exists(latest_link):
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    else:
        import shutil
        shutil.rmtree(latest_link)
os.symlink(os.path.basename(out_dir), latest_link, target_is_directory=True)

print(f"\n真实人类 vs 狗 VAD 对比完成！")
print(f"结果保存在: {out_dir}")
print(f"快捷访问: experiments/comparison/latest/")
print("\n对比表格：")
print(df.round(4))