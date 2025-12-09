# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:21
# @Site : 
# @file : train.py
# @Software : PyCharm
# @Description :

# train.py —— 每个 epoch 实时保存日志（支持随时中断）
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from datetime import datetime

from src.dataset import PairedDogDataset
from src.model import DogEmotionModel
from src.losses import BiProtoAlignLoss, EmergentLoss, EmergentVADLoss
from utils.tracker import PrototypeTracker
from utils.early_stopping import EarlyStopping


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_log(history, log_file):
    """实时保存日志（每个 epoch 调用一次）"""
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # 数据
    dataset = PairedDogDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=16, shuffle=False, num_workers=0)

    # 模型
    model = DogEmotionModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # 损失
    proto_loss = BiProtoAlignLoss(temp=0.07)
    emer_loss  = EmergentLoss()
    vad_loss   = EmergentVADLoss(lambda_r=2.0, lambda_o=0.2, delta=0.25)

    # 人类锚点
    human_anchors = torch.load("real_human_anchors.pt", map_location=device)
    human_anchors = human_anchors.detach()
    human_anchors.requires_grad = False

    tracker = PrototypeTracker(num_classes=8, dim=128, device=device)
    early_stop = EarlyStopping(patience=15, save_path="best_emergent_vad.pt")

    # 日志（实时保存）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_realtime_{timestamp}.json")

    history = {
        "timestamp": timestamp,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "best_val_loss": float('inf'),
        "best_epoch": 0,
        "epochs": []
    }
    print(f"实时日志将保存在: {log_file}")

    best_val = float('inf')

    try:
        for epoch in range(1, 101):
            model.train()
            p_meter = e_meter = v_meter = total_meter = AverageMeter()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch:>3}", colour="cyan"):
                optimizer.zero_grad()

                small_in = batch["small_dog"].to(device)
                large_in = batch["large_dog"].to(device)
                labels = batch["class_id"].to(device)

                s_emb, s_vad = model(small_in)
                l_emb, l_vad = model(large_in)

                emb = torch.cat([s_emb, l_emb], dim=0)
                vad = torch.cat([s_vad, l_vad], dim=0)
                lbl = torch.cat([labels, labels], dim=0)

                p_loss = proto_loss(emb, lbl, human_anchors, tracker.get_prototypes())
                e_loss = emer_loss(s_emb, l_emb)
                v_loss = vad_loss(vad, lbl, model.get_vad_weights())

                loss = 0.7 * p_loss + 1.0 * e_loss + 1.5 * v_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                tracker.update(emb.detach(), lbl)

                bs = labels.size(0)
                p_meter.update(p_loss.item(), bs)
                e_meter.update(e_loss.item(), bs)
                v_meter.update(v_loss.item(), bs)
                total_meter.update(loss.item(), bs)

            # 验证
            model.eval()
            val_total = AverageMeter()
            with torch.no_grad():
                for batch in val_loader:
                    small_in = batch["small_dog"].to(device)
                    large_in = batch["large_dog"].to(device)
                    labels = batch["class_id"].to(device)

                    s_emb, s_vad = model(small_in)
                    l_emb, l_vad = model(large_in)

                    emb = torch.cat([s_emb, l_emb], dim=0)
                    vad = torch.cat([s_vad, l_vad], dim=0)
                    lbl = torch.cat([labels, labels], dim=0)

                    p_loss = proto_loss(emb, lbl, human_anchors, tracker.get_prototypes())
                    e_loss = emer_loss(s_emb, l_emb)
                    v_loss = vad_loss(vad, lbl, model.get_vad_weights())

                    val_total.update((p_loss + 0.3 * e_loss + 1.0 * v_loss).item(), labels.size(0))

            # 记录当前 epoch
            epoch_log = {
                "epoch": epoch,
                "train_loss": round(total_meter.avg, 6),
                "train_proto": round(p_meter.avg, 6),
                "train_emer": round(e_meter.avg, 6),
                "train_vad": round(v_meter.avg, 6),
                "val_loss": round(val_total.avg, 6)
            }
            history["epochs"].append(epoch_log)

            # 每个 epoch 立即保存日志！
            if val_total.avg < best_val:
                best_val = val_total.avg
                history["best_val_loss"] = best_val
                history["best_epoch"] = epoch
                torch.save(model.state_dict(), "best_emergent_vad.pt")

            save_log(history, log_file)  # 实时保存！

            print(f"\n{'='*80}")
            print(f"Epoch {epoch:>3} | Train {total_meter.avg:.4f} "
                  f"(P {p_meter.avg:.4f} E {e_meter.avg:.4f} V {v_meter.avg:.4f}) | "
                  f"Val {val_total.avg:.4f}", "← BEST" if val_total.avg < best_val else "")

            if early_stop(val_total.avg, model):
                print("Early stopping!")
                break

    except KeyboardInterrupt:
        print("\n手动停止训练！日志已实时保存！")
    finally:
        save_log(history, log_file)
        print(f"最终日志已保存: {log_file}")
        print("你可以随时中断，数据永不丢失！")


if __name__ == '__main__':
    main()