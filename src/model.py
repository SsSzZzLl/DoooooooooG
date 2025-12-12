# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:20
# @Site : 
# @file : model.py
# @Software : PyCharm
# @Description : 

# src/model.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

class DogEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained("Ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.backbone.feature_extractor._freeze_parameters()

        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.vad_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128, 64)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(64, 3)),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x).last_hidden_state.mean(dim=1)
        emb = self.projector(features)
        vad = self.vad_head(emb)
        return emb, vad

    def get_vad_weights(self):
        return self.vad_head[2].weight