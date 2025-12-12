# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:54
# @Site : 
# @file : make_figures.py
# @Software : PyCharm
# @Description : 

# scripts/make_figures.py —— Windows 完全兼容 + 自动补缓存 + 0.5秒出图！
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from datetime import datetime
import shutil
from tqdm import tqdm

# ==================== 项目根目录 + 设备 ====================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 自动找所有实验 ====================
EXP_ROOT = os.path.join(ROOT, "experiments")
exp_dirs = [d for d in os.listdir(EXP_ROOT)
            if d.endswith("_full") or d.endswith("_no_vad") or d.endswith("_no_proto") or d.endswith("_random_anchor")]
exp_dirs = sorted(exp_dirs, reverse=True)

print(f"找到 {len(exp_dirs)} 个实验，开始生成图表...")

# ==================== 为每个实验生成缓存 + 5张图 ====================
for exp_dir_name in exp_dirs:
    exp_dir = os.path.join(EXP_ROOT, exp_dir_name)
    model_path = os.path.join(exp_dir, "best_model.pt")
    cache_path = os.path.join(exp_dir, "cache", "embeddings.npy")
    fig_dir = os.path.join(exp_dir, "figures")
    log_path = os.path.join(exp_dir, "training_log.json")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"跳过 {exp_dir_name}：未找到模型")
        continue

    print(f"\n正在处理: {exp_dir_name}")

    # 自动生成缓存（如果没有）
    if not os.path.exists(cache_path):
        print("  → 缓存未找到，自动生成...")
        from src.engine.inferer import Inferer

        inferer = Inferer(model_path, device, infer_batch_size=512)
        data = inferer.infer(cache_path)
    else:
        print(f"  → 命中缓存: {cache_path}")
        data = np.load(cache_path, allow_pickle=True).item()

    # 生成 5 张单模型图
    names = ["Anger", "Fear", "Alertness", "Anxiety", "Playfulness", "Happiness", "Discomfort", "Neutral"]
    colors = plt.cm.tab10(np.arange(8))

    # 1. 单色 PCA
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(data["small_emb"])
    plt.figure(figsize=(12, 10))
    plt.scatter(proj[:, 0], proj[:, 1], s=25, alpha=0.8, color='steelblue')
    plt.title("PCA (Monochrome)", fontsize=20)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_pca_monochrome.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # 2. 彩虹 PCA
    plt.figure(figsize=(12, 10))
    for i in range(8):
        mask = data["labels"] == i
        plt.scatter(proj[mask, 0], proj[mask, 1], s=25, alpha=0.8, label=names[i], color=colors[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("PCA (Colored)", fontsize=20)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_pca_colored.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # 3. VAD 雷达图
    mean_vad = np.array([data["vads"][data["labels"] == i].mean(0) for i in range(8)])
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    for i in range(8):
        values = np.concatenate([mean_vad[i], [mean_vad[i, 0]]])
        ax.plot(angles, values, 'o-', linewidth=4.5, label=names[i], color=colors[i])
        ax.fill(angles, values, alpha=0.2, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Valence", "Arousal", "Dominance"], fontsize=20)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=15)
    plt.title("Emergent VAD Space", fontsize=26, pad=50)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_vad_radar.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # 4. 训练曲线
    if os.path.exists(log_path):
        import json

        with open(log_path) as f:
            log = json.load(f)
        epochs = [e["epoch"] for e in log["epochs"]]
        train_loss = [e["train_loss"] for e in log["epochs"]]
        val_loss = [e["val_loss"] for e in log["epochs"]]
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_loss, 'o-', label="Train Loss", linewidth=3)
        plt.plot(epochs, val_loss, 's-', label="Val Loss", linewidth=3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "03_training_curve.png"), dpi=600, bbox_inches='tight')
        plt.close()

    # 5. 体型不变性
    cos_sim = np.sum(data["small_emb"] * data["large_emb"], axis=1) / (
            np.linalg.norm(data["small_emb"], axis=1) * np.linalg.norm(data["large_emb"], axis=1) + 1e-8)
    plt.figure(figsize=(11, 7))
    plt.hist(cos_sim, bins=60, range=(0.9, 1.0), color='purple', alpha=0.9)
    plt.axvline(cos_sim.mean(), color='red', linestyle='--', linewidth=4, label=f"Mean = {cos_sim.mean():.4f}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title(f"Size-Invariance\nMean = {cos_sim.mean():.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "05_size_invariance.png"), dpi=600, bbox_inches='tight')
    plt.close()

# ==================== 生成消融对比图（Windows 兼容！）
print("\n正在生成消融对比图...")
comp_dir = os.path.join(ROOT, "experiments", "comparison")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
this_run = os.path.join(comp_dir, timestamp)
os.makedirs(this_run, exist_ok=True)

# 收集缓存
cache_data = {}
for exp_dir_name in exp_dirs[:4]:
    cache_path = os.path.join(EXP_ROOT, exp_dir_name, "cache", "embeddings.npy")
    if os.path.exists(cache_path):
        name = exp_dir_name.split("_", 1)[1] if "_" in exp_dir_name else exp_dir_name
        cache_data[name] = np.load(cache_path, allow_pickle=True).item()

# 4格彩虹 PCA
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()
for idx, (name, data) in enumerate(cache_data.items()):
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(data["small_emb"])
    ax = axes[idx]
    for i in range(8):
        mask = data["labels"] == i
        ax.scatter(proj[mask, 0], proj[mask, 1], s=20, color=colors[i], alpha=0.7)
    ax.set_title(name, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle("Ablation Study: PCA (Colored)", fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(this_run, "pca_4panel_colored.png"), dpi=600, bbox_inches='tight')
plt.close()

# 4格单色 PCA
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()
for idx, (name, data) in enumerate(cache_data.items()):
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(data["small_emb"])
    ax = axes[idx]
    ax.scatter(proj[:, 0], proj[:, 1], s=20, color='steelblue', alpha=0.6)
    ax.set_title(name, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle("Ablation Study: PCA (Monochrome)", fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(this_run, "pca_4panel_monochrome.png"), dpi=600, bbox_inches='tight')
plt.close()

# Windows 兼容：直接复制文件夹
latest_link = os.path.join(comp_dir, "latest")
if os.path.exists(latest_link):
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    else:
        shutil.rmtree(latest_link)
shutil.copytree(this_run, latest_link)
print(f"消融对比图已生成 → {latest_link}")

print(f"\n所有任务完成！你现在拥有顶会一作全套图表！")