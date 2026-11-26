# Import the relevant libraries
import json
import os
import numpy as np

# Function to run cache path
def run_cache_path(run_index, seed, cache_dir):
    return os.path.join(cache_dir, f"run_{run_index}_seed_{seed}.json")

# Function to run model path
def run_model_path(run_index, seed, cache_dir):
    return os.path.join(cache_dir, f"run_{run_index}_seed_{seed}_model.pth")

# Function to load cached run
def load_cached_run(run_index, seed, cache_dir):
    path = run_cache_path(run_index, seed, cache_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# Function to make JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(value) for value in obj]
    return obj

# Function to save run cache
def save_run_cache(run_index, seed, result, cache_dir):
    path = run_cache_path(run_index, seed, cache_dir)
    tmp_path = path + ".tmp"
    serializable_result = make_json_serializable(result)

    with open(tmp_path, "w") as f:
        json.dump(serializable_result, f, indent=2)

    os.replace(tmp_path, path)
    print(f"[cache] Saved run {run_index} (seed={seed}) -> {path}")

# Avoid running same experiment: checking if results are already saved in a cache
def run_or_load(run_index, seed, cache_dir, run_single_experiment):
    cached = load_cached_run(run_index, seed, cache_dir)
    if cached is not None:
        print(f"[cache] Run {run_index} (seed={seed}) already cached — skipping training.")
        return cached

    result = run_single_experiment(run_index, seed)
    save_run_cache(run_index, seed, result, cache_dir)
    return result