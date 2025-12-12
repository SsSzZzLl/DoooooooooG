# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/10 下午10:55
# @Site : 
# @file : run.py
# @Software : PyCharm
# @Description : 

# run.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_full():
    os.system("python scripts/train_all.py full")

def train_no_vad():
    os.system("python scripts/train_all.py no_vad")

def train_no_proto():
    os.system("python scripts/train_all.py no_proto")

def train_random():
    os.system("python scripts/train_all.py random_anchor")

def resume():
    os.system("python scripts/train_all.py --resume")

def figures():
    os.system("python scripts/make_figures.py")

def vad():
    os.system("python scripts/compare_vad.py")


def all_in_one():
    """训练 full + 3个消融 + 出图 + VAD对比"""
    print("启动流程！（4个实验 + 出图 + VAD对比）")

    experiments = ["full", "no_vad", "no_proto", "random_anchor"]
    for exp in experiments:
        print(f"\n开始训练: {exp}")
        result = os.system(f"python scripts/train_all.py {exp}")
        if result != 0:
            print(f"{exp} 训练失败！跳过...")
        else:
            print(f"{exp} 训练完成！")

    print("\n所有实验训练完成！开始出图...")
    os.system("python scripts/make_figures.py")
    os.system("python scripts/compare_vad.py")
    print("\n全部完成！你现在拥有顶会一作全套结果！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", choices=["train","no_vad","no_proto","random","resume","figures","vad","all"])
    args = parser.parse_args()

    task_map = {
        "train": train_full,
        "no_vad": train_no_vad,
        "no_proto": train_no_proto,
        "random": train_random,
        "resume": resume,
        "figures": figures,
        "vad": vad,
        "all": all_in_one,
    }

    if args.task:
        task_map[args.task]()
    else:
        print("===")