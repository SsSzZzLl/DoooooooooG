# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:21
# @Site : 
# @file : train.py
# @Software : PyCharm
# @Description :

# train.py —— 带完整日志记录 + 实时美观显示（支持 Windows）
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from src.dataset import PairedDogDataset
from src.model import DogEmotionModel
from src.losses import BiProtoAlignLoss, EmergentLoss
from utils.tracker import PrototypeTracker
from utils.early_stopping import EarlyStopping


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ==================== 数据 ====================
    dataset = PairedDogDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    # ==================== 模型 & 优化器 ====================
    model = DogEmotionModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # ==================== 损失 & 锚点 ====================
    proto_loss = BiProtoAlignLoss(temp=0.07)
    emer_loss = EmergentLoss()

    human_anchors = torch.load("real_human_anchors.pt", map_location=device)
    human_anchors = human_anchors.detach()
    human_anchors.requires_grad = False
    print(f"已加载真实人类锚点: {human_anchors.shape}")

    tracker = PrototypeTracker(num_classes=8, dim=128, device=device)
    early_stop = EarlyStopping(patience=12, save_path="best_dog_model.pt")

    # ==================== 日志设置 ====================
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.json")

    # 用于保存每个 epoch 的日志
    history = {
        "timestamp": timestamp,
        "epochs": [],
        "best_val_loss": float('inf'),
        "train_samples": len(train_set),
        "val_samples": len(val_set)
    }

    best_val = float('inf')

    # ==================== 训练循环 ====================
    for epoch in range(1, 101):
        model.train()
        p_loss_meter = AverageMeter()
        e_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:>3}/100 [Train]", leave=False, colour="#00FF00")
        for batch in progress_bar:
            optimizer.zero_grad()

            small_input = batch["small_dog"].to(device)
            large_input = batch["large_dog"].to(device)
            labels = batch["class_id"].to(device)

            s_emb, _ = model(small_input)
            l_emb, _ = model(large_input)

            emb = torch.cat([s_emb, l_emb], dim=0)
            lbl = torch.cat([labels, labels], dim=0)

            p_loss = proto_loss(emb, lbl, human_anchors, tracker.get_prototypes())
            e_loss = emer_loss(s_emb, l_emb)
            loss = p_loss + 0.7 * e_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tracker.update(emb.detach(), lbl)

            bs = labels.size(0)
            p_loss_meter.update(p_loss.item(), bs)
            e_loss_meter.update(e_loss.item(), bs)
            total_loss_meter.update(loss.item(), bs)

            progress_bar.set_postfix({
                "Total": f"{total_loss_meter.avg:.4f}",
                "Proto": f"{p_loss_meter.avg:.4f}",
                "Emer": f"{e_loss_meter.avg:.4f}"
            }, refresh=True)

        # ==================== 验证 ====================
        model.eval()
        val_total = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                small_input = batch["small_dog"].to(device)
                large_input = batch["large_dog"].to(device)
                labels = batch["class_id"].to(device)

                s_emb, _ = model(small_input)
                l_emb, _ = model(large_input)
                emb = torch.cat([s_emb, l_emb], dim=0)
                lbl = torch.cat([labels, labels], dim=0)

                p_loss = proto_loss(emb, lbl, human_anchors, tracker.get_prototypes())
                e_loss = emer_loss(s_emb, l_emb)
                val_total.update((p_loss + 0.7 * e_loss).item(), labels.size(0))

        # ==================== 记录 & 打印 ====================
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(total_loss_meter.avg, 6),
            "train_proto": round(p_loss_meter.avg, 6),
            "train_emer": round(e_loss_meter.avg, 6),
            "val_loss": round(val_total.avg, 6)
        }
        history["epochs"].append(epoch_log)

        is_best = val_total.avg < best_val
        if is_best:
            best_val = val_total.avg
            history["best_val_loss"] = best_val
            history["best_epoch"] = epoch

        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch:>3} | "
              f"Train: {total_loss_meter.avg:>7.4f} "
              f"(Proto {p_loss_meter.avg:>6.4f} + Emer {e_loss_meter.avg:>6.4f}) | "
              f"Val: {val_total.avg:>7.4f} ", end="")
        if is_best:
            print("← BEST", end="")
        print()

        if early_stop(val_total.avg, model):
            print("Early stopping triggered!")
            break

    # ==================== 保存完整日志 ====================
    history["total_epochs"] = epoch
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n训练完成！")
    print(f"最佳 Val Loss: {best_val:.6f} @ Epoch {history.get('best_epoch', epoch)}")
    print(f"完整训练日志已保存至: {log_file}")
    print("现在可以运行 test_visualize.py 画图投稿了！")


if __name__ == '__main__':
    main()