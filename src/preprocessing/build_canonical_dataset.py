import argparse
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
from pathlib import Path

# --- Add this right below your imports if it isn't there already! ---
ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", default=str(ROOT / "data" / "embeddings" / "chemberta_embeddings.pkl"))
    parser.add_argument("--esm2", default=str(ROOT / "data" / "embeddings" / "esm2_embeddings.pkl"))
    parser.add_argument("--output_dir", default=str(ROOT / "data" / "canonical"))
    return parser.parse_args()

def process_file(csv_path, out_prefix, chem, esm, chem_dim, esm_dim):
    print(f"\nProcessing {csv_path}...")
    df = pd.read_csv(csv_path)
    num_rows = len(df)
    feature_dim = (chem_dim * 2) + (esm_dim * 2)
    
    X_out_path = f"{out_prefix}_X.npy"
    y_out_path = f"{out_prefix}_y.npy"
    X_mmap = np.lib.format.open_memmap(X_out_path, mode='w+', dtype=np.float32, shape=(num_rows, feature_dim))
    y_mmap = np.lib.format.open_memmap(y_out_path, mode='w+', dtype=np.int32, shape=(num_rows,))

    batch_size = 10000
    X_batch = []
    y_batch = []
    current_idx = 0

    for _, row in tqdm(df.iterrows(), total=num_rows):
        d1, d2 = row["drug_A"], row["drug_B"]

        # Get features (use zeros if missing)
        vec1 = np.concatenate([chem.get(d1, np.zeros(chem_dim)), esm.get(d1, np.zeros(esm_dim))])
        vec2 = np.concatenate([chem.get(d2, np.zeros(chem_dim)), esm.get(d2, np.zeros(esm_dim))])
        
        X_batch.append(np.concatenate([vec1, vec2]))
        y_batch.append(row["label"])

        if len(X_batch) == batch_size:
            X_mmap[current_idx : current_idx + batch_size] = X_batch
            y_mmap[current_idx : current_idx + batch_size] = y_batch
            current_idx += batch_size
            X_batch = []
            y_batch = []

    if len(X_batch) > 0:
        X_mmap[current_idx:] = X_batch
        y_mmap[current_idx:] = y_batch

    X_mmap.flush()
    y_mmap.flush()
    
    print(f"Successfully saved to disk: Shape {X_mmap.shape}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading embeddings...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))

    chem_dim = len(next(iter(chem.values())))
    esm_dim = len(next(iter(esm.values())))
    print(f"Feature Dimensions - ChemBERTa: {chem_dim}, ESM2: {esm_dim}. Total per pair: {(chem_dim*2) + (esm_dim*2)}")

    files_to_process = [
        ("train_S0.csv", f"{args.output_dir}/S0_train"),
        ("test_S0.csv", f"{args.output_dir}/S0_test"),
        ("train_cold.csv", f"{args.output_dir}/cold_train"),
        ("test_cold_S1.csv", f"{args.output_dir}/cold_test_S1"),
        ("test_cold_S2.csv", f"{args.output_dir}/cold_test_S2")
    ]

    for csv_file, out_prefix in files_to_process:
        if os.path.exists(csv_file):
            process_file(csv_file, out_prefix, chem, esm, chem_dim, esm_dim)
        else:
            print(f"Warning: {csv_file} not found. Skipping.")

if __name__ == "__main__":
    main()