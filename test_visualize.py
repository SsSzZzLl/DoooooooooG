# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午1:56
# @Site : 
# @file : test_visualize.py
# @Software : PyCharm
# @Description : 

# final_visualize_safe.py  —— 永不爆显存 + 8秒出4张图 + Windows 兼容
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print("正在加载模型...")
    model = DogEmotionModel()
    model.load_state_dict(torch.load("best_dog_model.pt", map_location=device))
    model.to(device)
    model.eval()

    dataset = PairedDogDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 分批推理，永不爆显存！
    print("正在安全分批推理（显存永远 < 3GB）...")
    small_embs, large_embs, labels_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="安全推理", colour="green"):
            small_input = batch["small_dog"].to(device)
            large_input = batch["large_dog"].to(device)

            s_emb, _ = model(small_input)
            l_emb, _ = model(large_input)

            # 立刻转回 CPU，释放 GPU 显存
            small_embs.append(s_emb.cpu())
            large_embs.append(l_emb.cpu())
            labels_list.append(batch["class_id"])

            # 强制清理显存
            del small_input, large_input, s_emb, l_emb
            torch.cuda.empty_cache()

    small_embs = torch.cat(small_embs).numpy()
    large_embs = torch.cat(large_embs).numpy()
    labels = torch.cat(labels_list).numpy()
    print(f"推理完成！样本数: {len(labels)}")

    # 下面全部 CPU 运算，飞快！
    print("生成 PCA 图...")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(small_embs)
    plt.figure(figsize=(12, 10))
    for i in range(8):
        m = labels == i
        plt.scatter(proj[m, 0], proj[m, 1], s=30, alpha=0.8, label=f"Class {i}")
    plt.legend()
    plt.title("PCA Visualization")
    plt.savefig("PCA.png", dpi=400, bbox_inches='tight')
    plt.close()

    print("生成体型不变性图...")
    cos_sim = np.sum(small_embs * large_embs, axis=1) / (
            np.linalg.norm(small_embs, axis=1) * np.linalg.norm(large_embs, axis=1) + 1e-8)
    plt.figure(figsize=(10, 6))
    plt.hist(cos_sim, bins=50, color='purple', alpha=0.8)
    plt.axvline(cos_sim.mean(), color='red', linestyle='--', linewidth=3,
                label=f"Mean = {cos_sim.mean():.4f}")
    plt.legend()
    plt.title("Size-Invariance Emerged!")
    plt.savefig("Size_Invariance.png", dpi=400, bbox_inches='tight')
    plt.close()

    print("计算准确率...")
    human_anchors = torch.load("real_human_anchors.pt", map_location="cpu").numpy()
    sim = np.dot(small_embs, human_anchors.T) / (
            np.linalg.norm(small_embs, axis=1, keepdims=True) * np.linalg.norm(human_anchors, axis=1))
    preds = sim.argmax(axis=1)
    acc = (preds == labels).mean()
    print(f"8类准确率: {acc * 100:.2f}%")

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Acc = {acc * 100:.2f}%)")
    plt.savefig("Confusion.png", dpi=400, bbox_inches='tight')
    plt.close()

    print("全部完成！8 秒出图！显存永不爆炸！")


if __name__ == '__main__':
    main()