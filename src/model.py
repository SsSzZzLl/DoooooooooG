# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:20
# @Site : 
# @file : model.py
# @Software : PyCharm
# @Description : 

# src/model.py —— 终极顶会版（支持 EmergentVAD Orthogonality）
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class DogEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 真实人类8类情感预训练模型
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
        self.regressor = nn.Linear(128, 3)  # VAD 输出

    def forward(self, x):
        out = self.backbone(x).last_hidden_state      # [B, T, 1024]
        pooled = out.mean(dim=1)                      # [B, 1024]
        emb = self.projector(pooled)                  # [B, 128]
        vad = torch.sigmoid(self.regressor(emb))      # [B, 3]
        return emb, vad

    # ==================== 关键修改：暴露回归头权重 ====================
    def get_vad_weights(self):
        """
        返回 VAD 回归头的权重矩阵 w_j, w_k ∈ ℝ^{128}
        用于 EmergentVAD Loss 的 Orthogonality 项：λ_o Σ |cos(w_j, w_k)|
        """
        return self.regressor.weight  # [3, 128]