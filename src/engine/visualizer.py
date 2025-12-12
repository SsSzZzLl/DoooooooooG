# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:52
# @Site : 
# @file : visualizer.py
# @Software : PyCharm
# @Description :

# src/engine/visualizer.py  —— 终极完整版（支持缓存 + 4060 优化 + 所有图）
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import torch

class Visualizer:
    def __init__(self, exp_dir, device=None):
        self.exp_dir = exp_dir
        self.figures_dir = os.path.join(exp_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.device = device or torch.device("cpu")
        self.names = ["Anger", "Fear", "Alertness", "Anxiety",
                      "Playfulness", "Happiness", "Discomfort", "Neutral"]
        self.colors = plt.cm.tab10(np.arange(8))

    # ==================== 两个入口：从缓存版 + 传统版 ====================
    def draw_all(self, model_path, log_path, human_anchors_path):
        """传统方式：需要 model_path"""
        from src.engine.inferer import Inferer
        cache_path = os.path.join(self.exp_dir, "cache", "embeddings.npy")
        inferer = Inferer(model_path, self.device)
        data = inferer.infer(cache_path)
        self._draw_all_from_data(data, log_path, human_anchors_path)

    def draw_all_from_cache(self, cache_path, log_path, human_anchors_path=None):
        """最快方式：直接读缓存"""
        data = np.load(cache_path, allow_pickle=True).item()
        self._draw_all_from_data(data, log_path, human_anchors_path)

    # ==================== 核心绘图函数 ====================
    def _draw_all_from_data(self, data, log_path, human_anchors_path):
        self.draw_pca_monochrome(data)
        self.draw_pca_colored(data)
        self.draw_vad_radar(data)
        self.draw_training_curve(log_path)
        self.draw_confusion_matrix(data, human_anchors_path or "real_human_anchors.pt")
        self.draw_size_invariance(data)

    def draw_pca_monochrome(self, data):
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(data["small_emb"])
        plt.figure(figsize=(12, 10))
        plt.scatter(proj[:, 0], proj[:, 1], s=25, alpha=0.8, color='steelblue')
        plt.title("PCA of Canine Emotion Embeddings (Monochrome)", fontsize=20)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "01_pca_monochrome.png"), dpi=600, bbox_inches='tight')
        plt.close()

    def draw_pca_colored(self, data):
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(data["small_emb"])
        plt.figure(figsize=(12, 10))
        for i in range(8):
            mask = data["labels"] == i
            plt.scatter(proj[mask, 0], proj[mask, 1], s=25, alpha=0.8,
                        label=self.names[i], color=self.colors[i])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.title("PCA of Canine Emotion Embeddings (Colored)", fontsize=20)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "01_pca_colored.png"), dpi=600, bbox_inches='tight')
        plt.close()

    def draw_vad_radar(self, data):
        mean_vad = np.array([data["vads"][data["labels"] == i].mean(0) for i in range(8)])
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        for i in range(8):
            values = np.concatenate([mean_vad[i], [mean_vad[i, 0]]])
            ax.plot(angles, values, 'o-', linewidth=4.5, label=self.names[i], color=self.colors[i])
            ax.fill(angles, values, alpha=0.2, color=self.colors[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Valence", "Arousal", "Dominance"], fontsize=20)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, linewidth=2)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=15)
        plt.title("Emergent VAD Space in Canine Vocalizations\n(Zero-Shot Cross-Species)",
                  fontsize=26, pad=50)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "02_vad_radar.png"), dpi=600, bbox_inches='tight')
        plt.close()

    def draw_training_curve(self, log_path):
        if not os.path.exists(log_path):
            print(f"日志未找到: {log_path}")
            return
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
        epochs = [e["epoch"] for e in log["epochs"]]
        train_loss = [e["train_loss"] for e in log["epochs"]]
        val_loss = [e["val_loss"] for e in log["epochs"]]
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_loss, 'o-', label="Train Loss", linewidth=3, color='tab:blue')
        plt.plot(epochs, val_loss, 's-', label="Val Loss", linewidth=3, color='tab:orange')
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Training and Validation Loss Curves", fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "03_training_curve.png"), dpi=600, bbox_inches='tight')
        plt.close()

    def draw_confusion_matrix(self, data, human_anchors_path):
        human_anchors = torch.load(human_anchors_path, map_location="cpu")
        sim = torch.cosine_similarity(
            torch.tensor(data["small_emb"]).unsqueeze(1),
            human_anchors.unsqueeze(0), dim=-1
        )
        pred = sim.argmax(1).numpy()
        acc = (pred == data["labels"]).mean()
        cm = confusion_matrix(data["labels"], pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.names, yticklabels=self.names,
                    cbar=False, linewidths=0.5, linecolor='gray')
        plt.title(f"Confusion Matrix (Acc = {acc*100:.2f}%)", fontsize=20)
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("True", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "04_confusion_matrix.png"), dpi=600, bbox_inches='tight')
        plt.close()

    def draw_size_invariance(self, data):
        cos_sim = np.sum(data["small_emb"] * data["large_emb"], axis=1) / (
            np.linalg.norm(data["small_emb"], axis=1) *
            np.linalg.norm(data["large_emb"], axis=1) + 1e-8
        )
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
        plt.savefig(os.path.join(self.figures_dir, "05_size_invariance.png"), dpi=600, bbox_inches='tight')
        plt.close()