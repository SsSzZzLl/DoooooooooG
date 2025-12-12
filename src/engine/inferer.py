# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:51
# @Site : 
# @file : inferer.py
# @Software : PyCharm
# @Description : 

# src/engine/inferer.py  —— 终极修复版（缓存必开！）
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.model import DogEmotionModel
from src.dataset import PairedDogDataset

class Inferer:
    def __init__(self, model_path, device, infer_batch_size=512):
        self.model = DogEmotionModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.infer_batch_size = infer_batch_size

    def infer(self, cache_path):
        if os.path.exists(cache_path):
            print(f"命中缓存！加载: {cache_path}")
            return np.load(cache_path, allow_pickle=True).item()

        print(f"缓存未命中，开始推理 → {cache_path}")
        dataset = PairedDogDataset()
        loader = DataLoader(
            dataset,
            batch_size=self.infer_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        small_embs, large_embs, vads, labels = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="推理进度", colour="green"):
                batch["small_dog"] = batch["small_dog"].to(self.device)
                batch["large_dog"] = batch["large_dog"].to(self.device)

                s_emb, s_vad = self.model(batch["small_dog"])
                l_emb, _ = self.model(batch["large_dog"])

                small_embs.append(s_emb.cpu().numpy())
                large_embs.append(l_emb.cpu().numpy())
                vads.append(s_vad.cpu().numpy())
                labels.append(batch["class_id"].numpy())

        data = {
            "small_emb": np.concatenate(small_embs),
            "large_emb": np.concatenate(large_embs),
            "vads": np.concatenate(vads),
            "labels": np.concatenate(labels)
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, data)
        print(f"缓存已保存: {cache_path}")
        return data