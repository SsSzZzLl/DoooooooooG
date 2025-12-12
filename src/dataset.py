# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:20
# @Site : 
# @file : dataset.py
# @Software : PyCharm
# @Description : 

# src/dataset.py
import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

class PairedDogDataset(Dataset):
    def __init__(self):
        # 正确计算项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src
        parent_dir = os.path.dirname(current_dir)  # E:\DoooooooooG
        self.original_root = os.path.join(parent_dir, "data", "Mescalina_2017")
        self.augmented_root = os.path.join(parent_dir, "data", "Mescalina_2017_augmented")  # ← 去掉空格！

        print(f"原始数据目录: {self.original_root}")
        print(f"增强数据目录: {self.augmented_root}")

        if not os.path.exists(self.original_root):
            raise FileNotFoundError(f"原始数据目录不存在！请放到: {self.original_root}")
        if not os.path.exists(self.augmented_root):
            print("检测到增强数据，将使用增强版本")
        else:
            print("未检测到增强数据，只使用原始数据")

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.max_len = 16000 * 3

        self.context_map = {
            'L-S2': 0, 'L-A': 0, 'L-S3': 1, 'CH-N': 1, 'L-S1': 2, 'L-PA': 2,
            'L-TA': 3, 'L-P': 4, 'CH-P': 4, 'L-H': 5, 'GR-P': 5, 'GR-N': 6, 'L-O': 7
        }

        self.files = []
        self.labels = []
        for dog in os.listdir(self.original_root):
            dog_path = os.path.join(self.original_root, dog)
            if not os.path.isdir(dog_path): continue
            for ctx in os.listdir(dog_path):
                if ctx not in self.context_map: continue
                ctx_path = os.path.join(dog_path, ctx)
                if not os.path.isdir(ctx_path): continue
                for f in os.listdir(ctx_path):
                    if f.lower().endswith(('.wav', '.mp3', '.flac')):
                        rel = os.path.join(dog, ctx, f)
                        self.files.append(rel)
                        self.labels.append(self.context_map[ctx])
        print(f"Loaded {len(self.files)} samples")

    def _load(self, path):
        try:
            y, _ = librosa.load(path, sr=16000)
        except:
            y = np.zeros(16000)
        if len(y) > self.max_len:
            y = y[:self.max_len]
        else:
            y = np.pad(y, (0, self.max_len - len(y)))
        input_values = self.processor(y, sampling_rate=16000, return_tensors="pt").input_values
        return input_values.squeeze(0)

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        rel = self.files[idx]
        orig = os.path.join(self.original_root, rel)
        aug = os.path.join(self.augmented_root, rel)
        if not os.path.exists(aug):
            aug = orig
        return {
            "small_dog": self._load(orig),
            "large_dog": self._load(aug),
            "class_id": torch.tensor(self.labels[idx], dtype=torch.long)
        }