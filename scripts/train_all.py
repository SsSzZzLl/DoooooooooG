# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:54
# @Site : 
# @file : train_all.py
# @Software : PyCharm
# @Description : 

# scripts/train_all.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from src.engine.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("exp_name", choices=["full", "no_vad", "no_proto", "random_anchor"])
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# 读取配置
with open("config/experiments.yaml", "r", encoding="utf-8") as f:
    configs = yaml.safe_load(f)

config = configs[args.exp_name]
config.update(configs["global"])

print(f"开始训练实验: {args.exp_name}")

trainer = Trainer(config, args.exp_name)
trainer.train(resume=args.resume)