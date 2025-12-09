# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:20
# @Site : 
# @file : model.py
# @Software : PyCharm
# @Description : 

# src/model.py
import torch.nn as nn
from sympy.printing.pytorch import torch
from transformers import Wav2Vec2Model

class DogEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(
            "Ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        self.backbone.feature_extractor._freeze_parameters()
        hidden = self.backbone.config.hidden_size  # 1024

        self.projector = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128)
        )
        # 顶级强制正交！
        self.regressor = nn.utils.spectral_norm(nn.Linear(128, 3))

    def forward(self, x):
        out = self.backbone(x).last_hidden_state
        pooled = out.mean(dim=1)
        emb = self.projector(pooled)
        vad = torch.sigmoid(self.regressor(emb))
        return emb, vad

    def get_vad_weights(self):
        return self.regressor.weight  # [3, 128]