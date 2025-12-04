import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

# =================CONFIGURATION=================
INPUT_ROOT = "data"
OUTPUT_ROOT = "data_augmented"
EXTENSIONS = {'.wav', '.mp3', '.flac'}  # Set is faster for lookup

# Simulation Parameters
PITCH_SHIFT_MIN = -5.5
PITCH_SHIFT_MAX = -2.5

# Number of CPU cores to use (-1 uses all available)
NUM_JOBS = -1


# ===============================================

def bio_acoustic_simulation(y, sr):
    """
    Simulates a larger dog by lowering the pitch while preserving time.
    """
    # Randomize size factor
    n_steps = np.random.uniform(PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)

    # Pitch shift (Time-invariant)
    y_large = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Add slight noise (Regularization)
    noise_amp = 0.005 * np.random.uniform() * np.amax(y_large)
    y_large = y_large + noise_amp * np.random.normal(size=y_large.shape[0])

    return y_large


def process_single_file(file_info):
    """
    Worker function for parallel processing.
    """
    input_path, output_path = file_info

    try:
        # optimization: Check if output already exists to allow resuming
        if os.path.exists(output_path):
            return 0  # Skip

        # Load original
        y, sr = librosa.load(input_path, sr=None)

        # Augment
        y_augmented = bio_acoustic_simulation(y, sr)

        # Save
        sf.write(output_path, y_augmented, sr)
        return 1  # Success
    except Exception as e:
        print(f"\nError processing {input_path}: {e}")
        return 0  # Fail


def process_dataset():
    print(f"Starting Bio-Acoustic Simulation (Parallel Mode)...")
    print(f"Input: {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")

    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPU cores.")

    # 1. Build the list of files to process first
    files_to_process = []

    print("Scanning directory structure...")
    for root, dirs, files in os.walk(INPUT_ROOT):
        rel_path = os.path.relpath(root, INPUT_ROOT)

        # Robust check: Ensure "Mescalina 2015" is not in any part of the path
        path_parts = rel_path.split(os.sep)
        if "Mescalina 2015" in path_parts:
            continue

        # Create output directory structure immediately
        output_dir = os.path.join(OUTPUT_ROOT, rel_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for file in files:
            # Check extension
            _, ext = os.path.splitext(file)
            if ext.lower() in EXTENSIONS:
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                files_to_process.append((input_path, output_path))

    total_files = len(files_to_process)
    print(f"Total audio files to process: {total_files}")

    # 2. Process in Parallel
    # We use tqdm to wrap the Parallel iterator to show progress
    results = Parallel(n_jobs=NUM_JOBS)(
        delayed(process_single_file)(info)
        for info in tqdm(files_to_process, desc="Augmenting Audio")
    )

    processed_count = sum(results)

    print("\nSimulation Complete.")
    print(f"Successfully processed {processed_count} / {total_files} files")
    print(f"Augmented dataset created at: {os.path.abspath(OUTPUT_ROOT)}")


if __name__ == "__main__":
    # Ensure input directory exists
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input directory '{INPUT_ROOT}' not found.")
        # Create dummy structure for testing
        os.makedirs("data/BarkDb/Fido/Aggressive", exist_ok=True)
        dummy_wav = np.zeros(22050)
        sf.write("data/BarkDb/Fido/Aggressive/bark1.wav", dummy_wav, 22050)

    process_dataset()