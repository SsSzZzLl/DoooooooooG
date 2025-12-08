# Author : SsSzZzLl
# -*- coding = utf-8 -*-
# @Time : 2025/12/7 下午11:29
# @Site : 
# @file : augmentation.py
# @Software : PyCharm
# @Description :

# augmentation.py
import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

INPUT_ROOT = "../data/Mescalina 2017"
OUTPUT_ROOT = "../data/Mescalina 2017_augmented"
EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg')
PITCH_SHIFT_MIN = -4.5
PITCH_SHIFT_MAX = -2.0
NOISE_FACTOR = 0.005
NUM_JOBS = -1  # 使用全部核心

def bio_acoustic_simulation(y, sr):
    n_steps = np.random.uniform(PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)
    try:
        y_large = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, res_type='kaiser_fast')
    except:
        y_large = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    if NOISE_FACTOR > 0:
        noise = NOISE_FACTOR * np.random.uniform() * np.amax(np.abs(y_large))
        y_large += noise * np.random.randn(len(y_large))
    return y_large

def process_file(input_path, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return 0
    try:
        y, sr = librosa.load(input_path, sr=None)
        if len(y) < sr * 0.1:
            return 0
        y_aug = bio_acoustic_simulation(y, sr)
        sf.write(output_path, y_aug, sr)
        return 1
    except:
        return 0

def run_augmentation():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    tasks = []

    print("Scanning all dog folders...")
    for dog_folder in os.listdir(INPUT_ROOT):
        dog_path = os.path.join(INPUT_ROOT, dog_folder)
        if not os.path.isdir(dog_path):
            continue
        for context in os.listdir(dog_path):
            context_path = os.path.join(dog_path, context)
            if not os.path.isdir(context_path):
                continue
            out_context = os.path.join(OUTPUT_ROOT, dog_folder, context)
            os.makedirs(out_context, exist_ok=True)

            for file in os.listdir(context_path):
                if file.lower().endswith(EXTENSIONS):
                    src = os.path.join(context_path, file)
                    dst = os.path.join(out_context, file)
                    tasks.append((src, dst))

    print(f"Found {len(tasks)} files to augment (simulating large dogs)")
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_file)(src, dst) for src, dst in tqdm(tasks, desc="Augmenting")
    )
    print(f"Augmentation complete! {sum(results)} files processed.")


if __name__ == "__main__":
    run_augmentation()