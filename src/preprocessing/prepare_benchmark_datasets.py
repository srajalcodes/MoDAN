import pandas as pd
import numpy as np
import pickle
import random
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = ROOT / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

chemberta_path = EMBEDDING_DIR / "chemberta_embeddings.pkl"
biosnap_path = BENCHMARK_DIR / "ChCh-Miner_durgbank-chem-chem.tsv"
stanford_npz_path = BENCHMARK_DIR / "drugbank_v5_stanfordnlp.npz"

out_dir = ROOT / "data" / "benchmark_splits"

out_dir.mkdir(parents=True, exist_ok=True)

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def generate_benchmark_splits(edges_list, valid_drugs, dataset_name, out_dir):
    print(f"\n--- Processing {dataset_name} ---")
    
    valid_edges = []
    for d1, d2 in edges_list:
        if d1 in valid_drugs and d2 in valid_drugs:
            # Enforce symmetry (Alphabetical sort)
            pair = tuple(sorted([d1, d2]))
            valid_edges.append(pair)
            
    # Deduplicate
    valid_edges = list(set(valid_edges))
    print(f"Original edges extracted: {len(edges_list)}")
    print(f"Edges retained (both drugs in our embeddings): {len(valid_edges)}")
    
    if len(valid_edges) == 0:
        print("Error: No overlapping drugs found!")
        return

    # 2. Get unique drugs
    benchmark_drugs = list(set([d for pair in valid_edges for d in pair]))
    print(f"Unique drugs in {dataset_name}: {len(benchmark_drugs)}")

    # 3. Generate True Negatives (1:1 Ratio)
    positive_set = set(valid_edges)
    negatives = set()
    print("Generating True Negatives...")
    
    while len(negatives) < len(valid_edges):
        d1, d2 = random.sample(benchmark_drugs, 2)
        pair = tuple(sorted([d1, d2]))
        if pair not in positive_set and pair not in negatives:
            negatives.add(pair)
            
    # 4. Combine and Label
    pos_df = pd.DataFrame(valid_edges, columns=['drug_A', 'drug_B'])
    pos_df['label'] = 1
    neg_df = pd.DataFrame(list(negatives), columns=['drug_A', 'drug_B'])
    neg_df['label'] = 0
    
    df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Total {dataset_name} Canonical Dataset Size: {len(df)}")

    # 5. Create S0 (Random Split)
    os.makedirs(out_dir, exist_ok=True)
    train_s0, test_s0 = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_s0.to_csv(f"{out_dir}/{dataset_name}_train_S0.csv", index=False)
    test_s0.to_csv(f"{out_dir}/{dataset_name}_test_S0.csv", index=False)
    
    # 6. Create S1 & S2 (Cold Split)
    seen_drugs, unseen_drugs = train_test_split(benchmark_drugs, test_size=0.2, random_state=42)
    seen_set, unseen_set = set(seen_drugs), set(unseen_drugs)
    
    mask_train_cold = df['drug_A'].isin(seen_set) & df['drug_B'].isin(seen_set)
    mask_s2 = df['drug_A'].isin(unseen_set) & df['drug_B'].isin(unseen_set)
    mask_s1 = (
        (df['drug_A'].isin(seen_set) & df['drug_B'].isin(unseen_set)) | 
        (df['drug_A'].isin(unseen_set) & df['drug_B'].isin(seen_set))
    )
    
    df[mask_train_cold].to_csv(f"{out_dir}/{dataset_name}_train_cold.csv", index=False)
    df[mask_s1].to_csv(f"{out_dir}/{dataset_name}_test_cold_S1.csv", index=False)
    df[mask_s2].to_csv(f"{out_dir}/{dataset_name}_test_cold_S2.csv", index=False)
    
    print(f"Successfully generated splits in '{out_dir}/'")
    print(f"  -> Cold Train (Old-Old): {mask_train_cold.sum()}")
    print(f"  -> S1 Test (Old-New):    {mask_s1.sum()}")
    print(f"  -> S2 Test (New-New):    {mask_s2.sum()}")

def main():
    set_seeds(42)
 
    print("Loading valid drugs from ChemBERTa embeddings...")
    with open(chemberta_path, "rb") as f:
        chem = pickle.load(f)
    valid_drugs = set(chem.keys())
    print(f"Total valid drugs available: {len(valid_drugs)}")

    print("\nReading BIOSNAP TSV...")
    try:
        biosnap_df = pd.read_csv(biosnap_path, sep='\t')
        # BIOSNAP usually has columns like 'Drug1' and 'Drug2'
        col1, col2 = biosnap_df.columns[0], biosnap_df.columns[1]
        biosnap_edges = list(zip(biosnap_df[col1], biosnap_df[col2]))
        generate_benchmark_splits(biosnap_edges, valid_drugs, "BIOSNAP", out_dir)
    except Exception as e:
        print(f"Failed to process BIOSNAP: {e}")

    print("\nReading Stanford NPZ...")
    try:
        npz_data = np.load(stanford_npz_path, allow_pickle=True)
        print("NPZ internal keys found:", npz_data.files)
        
        for key in npz_data.files:
            data = npz_data[key]
            if isinstance(data, np.ndarray):
                print(f"  -> '{key}': shape {data.shape}")
            else:
                print(f"  -> '{key}': type {type(data)}")
                
    except Exception as e:
        print(f"Failed to process NPZ: {e}")

if __name__ == "__main__":
    main()