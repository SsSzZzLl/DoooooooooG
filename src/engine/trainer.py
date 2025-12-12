# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:51
# @Site : 
# @file : trainer.py
# @Software : PyCharm
# @Description : 

# src/engine/trainer.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from datetime import datetime
import torch.nn.functional as F

from src.model import DogEmotionModel
from src.dataset import PairedDogDataset
from src.losses import BiProtoAlignLoss, EmergentLoss, EmergentVADLoss
from utils.tracker import PrototypeTracker
from utils.early_stopping import EarlyStopping
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint


class AverageMeter:
    def __init__(self): self.reset()

    def reset(self): self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    def __init__(self, config, exp_name):
        self.config = config
        self.exp_name = exp_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiments/{self.timestamp}_{exp_name}"
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/cache", exist_ok=True)
        self.checkpoint_path = os.path.join(self.exp_dir, "checkpoint_latest.pt")
        self.best_model_path = os.path.join(self.exp_dir, "best_model.pt")
        self.final_model_path = os.path.join(self.exp_dir, "final_model.pt")
        self.log_file = os.path.join(self.exp_dir, "training_log.json")
        self.logger = Logger(self.log_file)

        self.model = DogEmotionModel().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-2)
        self.proto_loss_fn = BiProtoAlignLoss()
        self.emer_loss_fn = EmergentLoss()
        self.vad_loss_fn = EmergentVADLoss()

        if self.config['anchor_type'] == "random":
            self.human_anchors = torch.randn(8, 128, device=self.device)
            self.human_anchors = F.normalize(self.human_anchors, dim=1)
        else:
            self.human_anchors = torch.load("real_human_anchors.pt", map_location=self.device)
        self.human_anchors = self.human_anchors.detach()
        self.human_anchors.requires_grad = False

        self.tracker = PrototypeTracker(num_classes=8, dim=128, device=self.device)
        self.early_stop = EarlyStopping(patience=self.config['early_stop_patience'], save_path=self.best_model_path)
        self.dynamic_anchors = self.human_anchors.clone().detach()

        self.dataset = PairedDogDataset()
        train_set, val_set = random_split(self.dataset, [int(0.8 * len(self.dataset)),
                                                         len(self.dataset) - int(0.8 * len(self.dataset))],
                                          generator=torch.Generator().manual_seed(self.config['seed']))
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.config['val_batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    def train(self, resume=False):
        start_epoch = 1
        history = {"timestamp": datetime.now().isoformat(), "mode": self.exp_name, "epochs": []}
        best_val = float('inf')

        if resume and os.path.exists(self.checkpoint_path):
            checkpoint = load_checkpoint(self.checkpoint_path, self.model, self.optimizer, self.device)
            start_epoch = checkpoint["epoch"] + 1
            self.dynamic_anchors = checkpoint["dynamic_anchors"].to(self.device)
            self.tracker = checkpoint["tracker"]
            history = checkpoint["history"]
            best_val = checkpoint["best_val"]
            print(f"恢复训练从 epoch {start_epoch}")

        try:
            for epoch in range(start_epoch, self.config['max_epochs'] + 1):
                self.model.train()
                meters = {"total": AverageMeter(), "proto": AverageMeter(), "emer": AverageMeter(),
                          "vad": AverageMeter()}

                vad_weight = eval(self.config.get('vad_weight_str', "0.3 + 1.7 * (epoch / 50.0)"))
                proto_temp = max(0.07 * (0.98 ** epoch), 0.01)

                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch:>3} Train", colour="blue"):
                    self.optimizer.zero_grad()

                    s_in = batch["small_dog"].to(self.device)
                    l_in = batch["large_dog"].to(self.device)
                    labels = batch["class_id"].to(self.device)

                    s_emb, s_vad = self.model(s_in)
                    l_emb, l_vad = self.model(l_in)

                    emb = torch.cat([s_emb, l_emb], dim=0)
                    vad = torch.cat([s_vad, l_vad], dim=0)
                    lbl = torch.cat([labels, labels], dim=0)

                    p_loss = torch.tensor(0.0, device=self.device) if not self.config[
                        'use_proto'] else self.proto_loss_fn(emb, lbl, self.dynamic_anchors,
                                                             self.tracker.get_prototypes(), temp=proto_temp)
                    e_loss = self.emer_loss_fn(s_emb, l_emb)
                    v_loss = torch.tensor(0.0, device=self.device) if not self.config['use_vad'] else self.vad_loss_fn(
                        vad, lbl, self.model.get_vad_weights())

                    loss = p_loss + 1.0 * e_loss + vad_weight * v_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.tracker.update(emb.detach(), lbl)

                    for k, v in zip(["total", "proto", "emer", "vad"], [loss, p_loss, e_loss, v_loss]):
                        meters[k].update(v.item(), labels.size(0))

                # 更新动态锚点
                if epoch % 5 == 0:
                    with torch.no_grad():
                        current_proto = self.tracker.get_prototypes()
                        self.dynamic_anchors = 0.99 * self.dynamic_anchors + 0.01 * current_proto
                        self.dynamic_anchors = self.dynamic_anchors.detach()

                # 验证
                self.model.eval()
                val_total = AverageMeter()
                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc=f"Epoch {epoch:>3} Val", colour="yellow"):
                        s_in = batch["small_dog"].to(self.device)
                        l_in = batch["large_dog"].to(self.device)
                        labels = batch["class_id"].to(self.device)

                        s_emb, s_vad = self.model(s_in)
                        l_emb, l_vad = self.model(l_in)

                        emb = torch.cat([s_emb, l_emb], dim=0)
                        vad = torch.cat([s_vad, l_vad], dim=0)
                        lbl = torch.cat([labels, labels], dim=0)

                        p_loss = torch.tensor(0.0) if not self.config['use_proto'] else self.proto_loss_fn(emb, lbl,
                                                                                                           self.dynamic_anchors,
                                                                                                           self.tracker.get_prototypes(),
                                                                                                           temp=proto_temp)
                        e_loss = self.emer_loss_fn(s_emb, l_emb)
                        v_loss = torch.tensor(0.0) if not self.config['use_vad'] else self.vad_loss_fn(vad, lbl,
                                                                                                       self.model.get_vad_weights())

                        val_total.update((p_loss + 1.0 * e_loss + vad_weight * v_loss).item(), labels.size(0))

                # 日志
                epoch_log = {
                    "epoch": epoch,
                    "train_loss": round(meters['total'].avg, 6),
                    "proto": round(meters['proto'].avg, 6),
                    "emer": round(meters['emer'].avg, 6),
                    "vad": round(meters['vad'].avg, 6),
                    "val_loss": round(val_total.avg, 6),
                    "vad_weight": round(vad_weight, 3)
                }
                history["epochs"].append(epoch_log)
                self.logger.log(history)

                if val_total.avg < best_val:
                    best_val = val_total.avg
                    torch.save(self.model.state_dict(), self.best_model_path)
                    print(f"    BEST MODEL SAVED! Val={best_val:.6f}")

                print(f"Epoch {epoch:>3} | Train {meters['total'].avg:.4f} | Val {val_total.avg:.4f}")

                # 自动保存检查点
                save_checkpoint(self.exp_dir, epoch, self.model, self.optimizer, self.dynamic_anchors, self.tracker,
                                history, best_val)

                if self.early_stop(val_total.avg, self.model):
                    print("Early stopping triggered!")
                    break

        except KeyboardInterrupt:
            print("\n手动停止训练！")
        finally:
            torch.save(self.model.state_dict(), self.final_model_path)
            print(f"\n训练完成！")
            print(f"最佳模型: {self.best_model_path}")
            print(f"最终模型: {self.final_model_path}")
            print(f"日志: {self.log_file}")

            # 关键！训练完立刻生成缓存和图！
            print("开始生成推理缓存和所有图表...")
            from src.engine.inferer import Inferer
            from src.engine.visualizer import Visualizer

            inferer = Inferer(self.best_model_path, self.device, infer_batch_size=512)
            cache_path = os.path.join(self.exp_dir, "cache", "embeddings.npy")
            data = inferer.infer(cache_path)

            visualizer = Visualizer(self.exp_dir, self.device)
            visualizer.draw_all(self.best_model_path, self.log_file, "real_human_anchors.pt")
            print("缓存和图表生成完成！")