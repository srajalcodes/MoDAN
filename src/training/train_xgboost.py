import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import argparse
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--train_csv", default="train_cold.csv")
    parser.add_argument("--s1_csv", default="test_cold_S1.csv")
    parser.add_argument("--s2_csv", default="test_cold_S2.csv")
    return parser.parse_args()

def get_features_for_batch(df_chunk, chem, esm, chem_dim, esm_dim):
    X, y = [], []
    for _, row in df_chunk.iterrows():
        d1, d2 = row["drug_A"], row["drug_B"]
        
        # Look up embeddings (use zeros if missing)
        vec1 = np.concatenate([chem.get(d1, np.zeros(chem_dim)), esm.get(d1, np.zeros(esm_dim))])
        vec2 = np.concatenate([chem.get(d2, np.zeros(chem_dim)), esm.get(d2, np.zeros(esm_dim))])
        
        X.append(np.concatenate([vec1, vec2]))
        y.append(row["label"])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def evaluate(model, csv_path, chem, esm, chem_dim, esm_dim, name):
    print(f"\n--- Evaluating on {name} ---")
    all_preds = []
    all_true = []
    
    # Read the test set in chunks so we don't crash the RAM
    chunk_size = 10000
    total_chunks = sum(1 for _ in open(csv_path)) // chunk_size
    
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), total=total_chunks):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, chem_dim, esm_dim)
        dtest = xgb.DMatrix(X_batch)
        
        preds = model.predict(dtest)
        all_preds.extend(preds)
        all_true.extend(y_batch)
        
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    bin_preds = (all_preds > 0.5).astype(int)
    
    print(f"ROC-AUC:   {roc_auc_score(all_true, all_preds):.4f}")
    print(f"F1 Score:  {f1_score(all_true, bin_preds):.4f}")
    print(f"Precision: {precision_score(all_true, bin_preds):.4f}")
    print(f"Recall:    {recall_score(all_true, bin_preds):.4f}")

def main():
    args = parse_args()
    
    print("Loading embedding dictionaries into RAM...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    chem_dim = len(next(iter(chem.values())))
    esm_dim = len(next(iter(esm.values())))
    
    params = {
        "max_depth": 6,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1
    }
    
    model = None
    chunk_size = 10000
    
    # Calculate total chunks for the progress bar
    total_train_lines = sum(1 for _ in open(args.train_csv))
    total_chunks = total_train_lines // chunk_size
    
    print(f"\nTraining XGBoost incrementally on {args.train_csv} (Zero Hard Drive Space Used)...")
    
    # Read the CSV in pieces. Process a piece, update the model, throw the piece away.
    for chunk in tqdm(pd.read_csv(args.train_csv, chunksize=chunk_size), total=total_chunks):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, chem_dim, esm_dim)
        dtrain = xgb.DMatrix(X_batch, label=y_batch)
        
        # update the model with the new chunk
        model = xgb.train(params, dtrain, num_boost_round=10, xgb_model=model)
        
    print("\nTraining Complete! Now running evaluation...")
    
    # Evaluate S1 and S2
    evaluate(model, args.s1_csv, chem, esm, chem_dim, esm_dim, "S1 (Known-New)")
    evaluate(model, args.s2_csv, chem, esm, chem_dim, esm_dim, "S2 (New-New)")

if __name__ == "__main__":
    main()