# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/9 下午6:19
# @Site : 
# @file : ablation_runner.py
# @Software : PyCharm
# @Description : 

# ablation_smart.py —— 聪明消融：只跑3个，复用你已有的 Full Model
import os
import subprocess
from datetime import datetime

# ！！！重要：改成你已经跑好的完整模型路径！！！
FULL_MODEL_DIR = "runs/run_20251209_103208"  # ← 改成你的路径！！！

if not os.path.exists(os.path.join(FULL_MODEL_DIR, "best_model.pt")):
    raise FileNotFoundError(f"找不到你的完整模型！请检查路径: {FULL_MODEL_DIR}")

# 创建消融结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_root = f"ablation_results/smart_ablation_{timestamp}"
os.makedirs(result_root, exist_ok=True)

print("聪明消融实验启动！")
print(f"复用完整模型: {FULL_MODEL_DIR}")
print(f"只跑3个消融，保存至: {result_root}\n")

# 只跑这3个消融
ABLATIONS = [
    {"name": "no_vad", "desc": "去除 EmergentVAD 损失", "env": {"USE_VAD": "False"}},
    {"name": "no_proto", "desc": "去除 BiProtoAlign 损失", "env": {"USE_PROTO": "False"}},
    {"name": "random_anchor", "desc": "使用随机人类锚点", "env": {"ANCHOR_TYPE": "random"}},
]

for abl in ABLATIONS:
    run_dir = os.path.join(result_root, abl["name"])
    os.makedirs(run_dir, exist_ok=True)

    print(f"正在运行: {abl['desc']}")
    print(f"   保存目录: {run_dir}")

    env = os.environ.copy()
    env.update(abl["env"])
    env["RUN_DIR"] = run_dir
    env["ABLATION_MODE"] = abl["name"]

    # 运行训练（建议20个epoch就够看出崩盘效果）
    subprocess.run(["python", "train.py"], env=env)
    print(f"{abl['desc']} 完成！\n")

print(f"\n聪明消融全部完成！")
print(f"完整模型（复用）: {FULL_MODEL_DIR}")
print(f"3个消融保存在: {result_root}")
