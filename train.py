# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/8 上午12:21
# @Site : 
# @file : train.py
# @Software : PyCharm
# @Description :

# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from datetime import datetime
import torch.nn.functional as F

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # ==================== 消融实验支持（通过环境变量控制）===================
    ABLATION_MODE = os.getenv("ABLATION_MODE", "full")  # full / no_vad / no_proto / random_anchor
    RUN_DIR = os.getenv("RUN_DIR")  # 消融时强制指定保存目录
    VAD_WEIGHT_STR = os.getenv("VAD_WEIGHT", "0.3 + 1.7 * (epoch / 50.0)")
    USE_PROTO = os.getenv("USE_PROTO", "True") == "True"
    USE_VAD = os.getenv("USE_VAD", "True") == "True"
    ANCHOR_TYPE = os.getenv("ANCHOR_TYPE", "human")  # human / random

    print(f"【消融模式】: {ABLATION_MODE}")
    print(f"Proto: {USE_PROTO} | VAD: {USE_VAD} | Anchor: {ANCHOR_TYPE}")

    # ==================== 自动创建实验目录（支持强制指定 RUN_DIR）===================
    if RUN_DIR:
        run_dir = RUN_DIR
        os.makedirs(run_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/run_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)

    best_model_path = os.path.join(run_dir, "best_model.pt")
    final_model_path = os.path.join(run_dir, "final_model.pt")
    log_file = os.path.join(run_dir, "training_log.json")

    print(f"实验保存目录: {run_dir}\n")

    # ==================== 数据加载 ====================
    dataset = PairedDogDataset()
    train_set, val_set = random_split(
        dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    # ==================== 模型与优化器 ====================
    model = DogEmotionModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    proto_loss_fn = BiProtoAlignLoss()
    emer_loss_fn = EmergentLoss()
    vad_loss_fn = EmergentVADLoss(lambda_r=2.0, lambda_o=0.2, delta=0.25)

    # ==================== 人类锚点（支持随机锚点消融）===================
    if ANCHOR_TYPE == "random":
        print("警告: 使用随机人类锚点（消融实验）")
        human_anchors = torch.randn(8, 128, device=device)
        human_anchors = F.normalize(human_anchors, dim=1)
    else:
        human_anchors = torch.load("real_human_anchors.pt", map_location=device)

    human_anchors = human_anchors.detach()
    human_anchors.requires_grad = False

    tracker = PrototypeTracker(num_classes=8, dim=128, device=device)
    early_stop = EarlyStopping(patience=5, save_path=best_model_path)

    dynamic_anchors = human_anchors.clone().detach()

    history = {
        "timestamp": datetime.now().isoformat(),
        "mode": ABLATION_MODE,
        "epochs": []
    }
    best_val = float('inf')

    try:
        for epoch in range(1, 101):
            model.train()
            meters = {k: AverageMeter() for k in ["total", "proto", "emer", "vad"]}

            vad_weight = eval(VAD_WEIGHT_STR)
            proto_temp = max(0.07 * (0.98 ** epoch), 0.01)

            for batch in tqdm(train_loader, desc=f"Epoch {epoch:>3}", colour="cyan"):
                optimizer.zero_grad()

                s_in = batch["small_dog"].to(device)
                l_in = batch["large_dog"].to(device)
                labels = batch["class_id"].to(device)

                s_emb, s_vad = model(s_in)
                l_emb, l_vad = model(l_in)

                emb = torch.cat([s_emb, l_emb], dim=0)
                vad = torch.cat([s_vad, l_vad], dim=0)
                lbl = torch.cat([labels, labels], dim=0)

                # Proto Loss（可消融）
                p_loss = proto_loss_fn(emb, lbl, dynamic_anchors, tracker.get_prototypes(), temp=proto_temp) if USE_PROTO else torch.tensor(0.0, device=device)

                e_loss = emer_loss_fn(s_emb, l_emb)

                # VAD Loss（可消融）
                v_loss = vad_loss_fn(vad, lbl, model.get_vad_weights()) if USE_VAD else torch.tensor(0.0, device=device)

                loss = p_loss + 1.0 * e_loss + vad_weight * v_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                tracker.update(emb.detach(), lbl)

                for k, v in zip(["total","proto","emer","vad"], [loss, p_loss, e_loss, v_loss]):
                    meters[k].update(v.item(), labels.size(0))

            # 每5 epoch 更新动态锚点
            if epoch % 5 == 0:
                with torch.no_grad():
                    current_proto = tracker.get_prototypes()
                    dynamic_anchors = 0.99 * dynamic_anchors + 0.01 * current_proto
                    dynamic_anchors = dynamic_anchors.detach()

            # ==================== 验证 ====================
            model.eval()
            val_total = AverageMeter()
            with torch.no_grad():
                for batch in val_loader:
                    s_in = batch["small_dog"].to(device)
                    l_in = batch["large_dog"].to(device)
                    labels = batch["class_id"].to(device)

                    s_emb, s_vad = model(s_in)
                    l_emb, l_vad = model(l_in)

                    emb = torch.cat([s_emb, l_emb], dim=0)
                    vad = torch.cat([s_vad, l_vad], dim=0)
                    lbl = torch.cat([labels, labels], dim=0)

                    p_loss = proto_loss_fn(emb, lbl, dynamic_anchors, tracker.get_prototypes(), temp=proto_temp) if USE_PROTO else 0.0
                    e_loss = emer_loss_fn(s_emb, l_emb)
                    v_loss = vad_loss_fn(vad, lbl, model.get_vad_weights()) if USE_VAD else 0.0

                    val_total.update((p_loss + 1.0 * e_loss + vad_weight * v_loss).item(), labels.size(0))

            # ==================== 日志记录 ====================
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

            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            if val_total.avg < best_val:
                best_val = val_total.avg
                print(f"    BEST MODEL SAVED! Val={best_val:.6f}")

            print(f"Epoch {epoch:>3} | Train {meters['total'].avg:.4f} | Val {val_total.avg:.4f}")

            if early_stop(val_total.avg, model):
                print("Early stopping!")
                break

    except KeyboardInterrupt:
        print("\n手动停止训练！")
    finally:
        torch.save(model.state_dict(), final_model_path)
        print(f"\n训练完成！")
        print(f"最佳模型: {best_model_path}")
        print(f"最终模型: {final_model_path}")
        print(f"日志: {log_file}")

if __name__ == '__main__':
    main()