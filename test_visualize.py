# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午1:56
# @Site :
# @file : test_visualize.py
# @Software : PyCharm
# @Description :

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

print("=" * 80)
print("DoooooooooG 终极可视化神器启动！强制GPU版！")
print("=" * 80)


# ==================== 自动扫描所有模型路径 ====================
def find_all_models():
    models = []
    if os.path.exists("runs"):
        for d in os.listdir("runs"):
            if d.startswith("run_"):
                path = os.path.join("runs", d)
                if os.path.exists(os.path.join(path, "best_model.pt")):
                    models.append({"path": path, "name": "Full Model", "type": "single"})
    if os.path.exists("ablation_results"):
        for root, dirs, files in os.walk("ablation_results"):
            for d in dirs:
                sub_path = os.path.join(root, d)
                if os.path.exists(os.path.join(sub_path, "best_model.pt")):
                    name_map = {
                        "full_model": "Full Model",
                        "no_vad": "w/o EmergentVAD",
                        "no_proto": "w/o BiProtoAlign",
                        "random_anchor": "Random Anchors"
                    }
                    name = name_map.get(d, d.replace("_", " ").title())
                    models.append({"path": sub_path, "name": name, "type": "ablation"})
    return sorted(models, key=lambda x: os.path.getctime(x["path"]), reverse=True)


all_models = find_all_models()
if not all_models:
    raise FileNotFoundError("未找到任何 best_model.pt！")

print(f"找到 {len(all_models)} 个可用模型：")
for i, m in enumerate(all_models):
    print(f"  {i + 1:2d}. {m['name']:30s} → {m['path']}")

main_model = all_models[0]
main_path = main_model["path"]
main_vis_dir = os.path.join(main_path, "visualizations")
os.makedirs(main_vis_dir, exist_ok=True)

print(f"\n主可视化模型: {main_model['name']}")
print(f"所有图表将保存至: {main_vis_dir}\n")

# ==================== 强制使用 GPU！===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("警告：未检测到GPU！将使用CPU（会很慢！）")
else:
    print(f"使用GPU: {torch.cuda.get_device_name(0)} 显存: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3}GB")


# ==================== 修复版推理函数（强制走GPU！）===================
def infer_model_gpu(model_path, model_name="未知模型"):
    print(f"正在推理模型: {model_name}")
    start_time =time.time()

    # 关键修复1：模型加载后立刻 .to(device)
    model = DogEmotionModel()
    state_dict = torch.load(model_path, map_location=device)  # 直接加载到GPU！
    model.load_state_dict(state_dict)
    model.to(device)  # 强制移到GPU
    model.eval()

    dataset = PairedDogDataset()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    small_embs, large_embs, vads, labels = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理进度", unit="batch", colour="green"):
            batch["small_dog"] = batch["small_dog"].to(device)
            batch["large_dog"] = batch["large_dog"].to(device)

            s_emb, s_vad = model(batch["small_dog"])
            l_emb, _ = model(batch["large_dog"])

            small_embs.append(s_emb.cpu().numpy())
            large_embs.append(l_emb.cpu().numpy())
            vads.append(s_vad.cpu().numpy())
            labels.append(batch["class_id"].numpy())

    elapsed = time.time() - start_time
    print(f"推理完成！耗时: {elapsed:.1f}秒\n")

    return (
        np.concatenate(small_embs),
        np.concatenate(large_embs),
        np.concatenate(vads),
        np.concatenate(labels)
    )


# ==================== 主模型推理（现在走GPU了！）===================
print("开始推理主模型...")
small_emb, large_emb, vads, labels = infer_model_gpu(
    os.path.join(main_path, "best_model.pt"),
    model_name=main_model["name"]
)

names = ["Anger", "Fear", "Alertness", "Anxiety", "Playfulness", "Happiness", "Discomfort", "Neutral"]
colors = plt.cm.tab10(np.arange(8))

# ==================== 图1：PCA 可视化 ====================
print("1/12 生成 PCA 图...")
start = time.time()
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(small_emb)
plt.figure(figsize=(12, 10))
for i in range(8):
    mask = labels == i
    plt.scatter(proj[mask, 0], proj[mask, 1], s=25, alpha=0.8, label=names[i], color=colors[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.title("PCA of Canine Emotion Embeddings", fontsize=20)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "01_pca_collapse.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"PCA 图完成！耗时: {time.time() - start:.1f}秒")

# ==================== 图2：体型不变性 ====================
print("2/12 生成体型不变性图...")
start = time.time()
cos_sim = np.sum(small_emb * large_emb, axis=1) / (
        np.linalg.norm(small_emb, axis=1) * np.linalg.norm(large_emb, axis=1) + 1e-8)
plt.figure(figsize=(11, 7))
plt.hist(cos_sim, bins=60, range=(0.9, 1.0), color='purple', alpha=0.9, edgecolor='black', linewidth=0.3)
plt.axvline(cos_sim.mean(), color='red', linestyle='--', linewidth=4,
            label=f"Mean = {cos_sim.mean():.4f}")
plt.xlabel("Cosine Similarity (Small vs Large Dog)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title(f"Size-Invariance Emerged!\nMean Similarity = {cos_sim.mean():.4f}", fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "02_size_invariance.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"体型不变性图完成！耗时: {time.time() - start:.1f}秒")

# ==================== 图3：混淆矩阵 ====================
print("3/12 生成混淆矩阵...")
start = time.time()
human_anchors = torch.load("real_human_anchors.pt", map_location="cpu")
sim = torch.cosine_similarity(torch.tensor(small_emb).unsqueeze(1), human_anchors.unsqueeze(0), dim=-1)
pred = sim.argmax(1).numpy()
acc = (pred == labels).mean()
cm = confusion_matrix(labels, pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=names, yticklabels=names, cbar=False, linewidths=0.5)
plt.title(f"Confusion Matrix (Accuracy = {acc * 100:.2f}%)", fontsize=20)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "03_confusion_matrix.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"混淆矩阵完成！准确率: {acc * 100:.2f}% 耗时: {time.time() - start:.1f}秒")

# ==================== 图4：VAD 雷达图 ====================
print("4/12 生成 VAD 雷达图...")
start = time.time()
mean_vad = np.array([vads[labels == i].mean(0) for i in range(8)])
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
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, linewidth=2)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=15)
plt.title("Emergent VAD Space in Canine Vocalizations\n(Zero-Shot Cross-Species)",
          fontsize=26, pad=50)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "04_vad_radar.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"VAD 雷达图完成！耗时: {time.time() - start:.1f}秒")

# ==================== 图5：3D VAD 空间 ====================
print("5/12 生成 3D VAD 空间图...")
start = time.time()
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(8):
    idx = labels == i
    ax.scatter(vads[idx, 0], vads[idx, 1], vads[idx, 2],
               c=[colors[i]], label=names[i], s=30, alpha=0.7)
ax.set_xlabel("Valence", fontsize=14)
ax.set_ylabel("Arousal", fontsize=14)
ax.set_zlabel("Dominance", fontsize=14)
ax.set_title("3D Emergent VAD Space\n(Zero-Shot Cross-Species)", fontsize=20)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "05_3d_vad_space.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"3D VAD 空间图完成！耗时: {time.time() - start:.1f}秒")

# ==================== 图6：VAD 箱型图 ====================
print("6/12 生成 VAD 箱型图...")
start = time.time()
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for i, title in enumerate(["Valence", "Arousal", "Dominance"]):
    data = [vads[labels == c, i] for c in range(8)]
    bp = axes[i].boxplot(data, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[i].set_title(title, fontsize=16)
    axes[i].tick_params(axis='x', rotation=45)
plt.suptitle("VAD Distribution per Emotion Class", fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "06_vad_boxplot.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"VAD 箱型图完成！耗时: {time.time() - start:.1f}秒")

# ==================== 图7：VAD 柱状图 ====================
print("7/12 生成 VAD 柱状图...")
start = time.time()
x = np.arange(8)
width = 0.25
fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x - width, mean_vad[:, 0], width, label="Valence", color='skyblue', alpha=0.9)
ax.bar(x, mean_vad[:, 1], width, label="Arousal", color='lightcoral', alpha=0.9)
ax.bar(x + width, mean_vad[:, 2], width, label="Dominance", color='lightgreen', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, fontsize=12)
ax.set_ylabel("Score", fontsize=14)
ax.legend(fontsize=14)
ax.set_title("Mean VAD Values per Emotion Class", fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "07_vad_bar_chart.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"VAD 柱状图完成！耗时: {time.time() - start:.1f}秒")

# ==================== 图8：类间 VAD 相似度热力图 ====================
print("8/12 生成类间 VAD 相似度热力图...")
start = time.time()
sim_matrix = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        sim_matrix[i, j] = np.dot(mean_vad[i], mean_vad[j]) / (
                np.linalg.norm(mean_vad[i]) * np.linalg.norm(mean_vad[j]) + 1e-8)
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="coolwarm",
            xticklabels=names, yticklabels=names, center=0)
plt.title("Inter-Class VAD Cosine Similarity", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(main_vis_dir, "08_vad_similarity_heatmap.png"), dpi=600, bbox_inches='tight')
plt.close()
print(f"类间相似度热力图完成！耗时: {time.time() - start:.1f}秒")

# ==================== 图9：训练曲线 ====================
print("9/12 生成训练曲线...")
start = time.time()
log_path = os.path.join(main_path, "training_log.json")
if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        log = json.load(f)
    if "epochs" in log:
        epochs = [e["epoch"] for e in log["epochs"]]
        train_loss = [e.get("train_loss", 0) for e in log["epochs"]]
        val_loss = [e.get("val_loss", 0) for e in log["epochs"]]
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_loss, 'o-', label="Train Loss", linewidth=3)
        plt.plot(epochs, val_loss, 's-', label="Val Loss", linewidth=3)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Training and Validation Loss Curves", fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(main_vis_dir, "09_training_curve.png"), dpi=600, bbox_inches='tight')
        plt.close()
        print(f"训练曲线完成！耗时: {time.time() - start:.1f}秒")
    else:
        print("日志中无 epochs 数据，跳过训练曲线")
else:
    print("未找到 training_log.json，跳过训练曲线")

# ==================== 消融对比图（如果有多个模型）===================
if len(all_models) > 1:
    print(f"\n检测到 {len(all_models)} 个模型，开始生成消融对比图...")

    # 图10：PCA 对比
    print("10/12 生成消融PCA对比图...")
    start = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for idx, m in enumerate(tqdm(all_models[:4], desc="消融PCA", leave=False)):
        emb, _, _, lab = infer_model_gpu(os.path.join(m["path"], "best_model.pt"), m["name"])
        proj = PCA(n_components=2, random_state=42).fit_transform(emb)
        ax = axes[idx]
        for i in range(8):
            mask = lab == i
            ax.scatter(proj[mask, 0], proj[mask, 1], s=20, alpha=0.7, color=colors[i])
        ax.set_title(m["name"], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.suptitle("Ablation Study: PCA Visualization", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(main_vis_dir, "10_ablation_pca.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"消融PCA对比图完成！耗时: {time.time() - start:.1f}秒")

    # 图11：VAD 雷达图对比
    print("11/12 生成消融VAD雷达图对比...")
    start = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    for idx, m in enumerate(tqdm(all_models[:4], desc="消融VAD雷达", leave=False)):
        _, _, vad, lab = infer_model_gpu(os.path.join(m["path"], "best_model.pt"), m["name"])
        mean_v = np.array([vad[lab == i].mean(0) for i in range(8)])
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        ax = axes[idx]
        for i in range(8):
            v = np.concatenate([mean_v[i], [mean_v[i, 0]]])
            ax.plot(angles, v, 'o-', linewidth=3, color=colors[i])
            ax.fill(angles, v, alpha=0.15, color=colors[i])
        ax.set_title(m["name"], fontsize=14)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["V", "A", "D"])
        ax.set_ylim(0, 1)
    plt.suptitle("Ablation Study: Emergent VAD Space", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(main_vis_dir, "11_ablation_vad_radar.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"消融VAD雷达图对比完成！耗时: {time.time() - start:.1f}秒")

    # 图12：准确率柱状图对比
    print("12/12 生成消融准确率柱状图...")
    start = time.time()
    accs = []
    names_list = []
    human_anchors = torch.load("real_human_anchors.pt", map_location="cpu")
    for m in tqdm(all_models, desc="计算准确率", leave=False):
        emb, _, _, lab = infer_model_gpu(os.path.join(m["path"], "best_model.pt"), m["name"])
        sim = torch.cosine_similarity(torch.tensor(emb).unsqueeze(1), human_anchors.unsqueeze(0), dim=-1)
        pred = sim.argmax(1).numpy()
        acc = (pred == lab).mean() * 100
        accs.append(acc)
        names_list.append(m["name"][:15])

    plt.figure(figsize=(12, 7))
    bars = plt.bar(names_list, accs, color=['gold'] + ['lightcoral'] * (len(accs) - 1), alpha=0.9)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Ablation Study: Classification Accuracy", fontsize=20)
    plt.xticks(rotation=15)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc:.2f}%",
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(main_vis_dir, "12_ablation_accuracy.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"消融准确率柱状图完成！耗时: {time.time() - start:.1f}秒")

print("\n" + "=" * 80)
print("所有12张图表生成完成！")
print(f"全部保存在：{main_vis_dir}")
print("=" * 80)
