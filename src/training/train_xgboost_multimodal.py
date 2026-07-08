import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
import pickle
import argparse
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import argparse
from pathlib import Path

# --- Add this right below your imports if it isn't there already! ---
ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", default="data/embeddings/chemberta_embeddings.pkl")
    parser.add_argument("--esm2", default="data/embeddings/esm2_embeddings.pkl")
    parser.add_argument("--biobert", default="data/embeddings/biobert_drug_embeddings.pkl")
    parser.add_argument("--train_csv", default="data/processed/train_cold.csv")
    parser.add_argument("--s1_csv", default="data/processed/test_cold_S1.csv")
    parser.add_argument("--s2_csv", default="data/processed/test_cold_S2.csv")
    parser.add_argument("--output", default="data/processed/multimodal_xgboost_results.csv")
    return parser.parse_args()

def get_features_for_batch(df_chunk, chem, esm, bio, chem_dim, esm_dim, bio_dim):
    X, y = [], []
    for _, row in df_chunk.iterrows():
        d1, d2 = row["drug_A"], row["drug_B"]
        
        # Drug A
        c1 = chem.get(d1, np.zeros(chem_dim))
        e1 = esm.get(d1, np.zeros(esm_dim))
        b1 = bio.get(d1, np.zeros(bio_dim))
        
        # Drug B
        c2 = chem.get(d2, np.zeros(chem_dim))
        e2 = esm.get(d2, np.zeros(esm_dim))
        b2 = bio.get(d2, np.zeros(bio_dim))
        
        # Concatenate: [c1, e1, b1, c2, e2, b2]
        vec1 = np.concatenate([c1, e1, b1])
        vec2 = np.concatenate([c2, e2, b2])
        X.append(np.concatenate([vec1, vec2]))
        y.append(row["label"])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def evaluate(xgb_model, lr_model, csv_path, chem, esm, bio, chem_dim, esm_dim, bio_dim, name):
    print(f"\n--- Evaluating on {name} ---")
    xgb_preds, lr_probs, all_true = [], [], []
    chunk_size = 10000
    
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size)):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, bio, chem_dim, esm_dim, bio_dim)
        
        # XGB Predict
        dtest = xgb.DMatrix(X_batch)
        xgb_preds.extend(xgb_model.predict(dtest))
        
        # LR Predict
        lr_probs.extend(lr_model.predict_proba(X_batch)[:, 1])
        all_true.extend(y_batch)
        
    xgb_preds = np.array(xgb_preds)
    lr_probs = np.array(lr_probs)
    all_true = np.array(all_true)
    
    xgb_bin = (xgb_preds > 0.5).astype(int)
    lr_bin = (lr_probs > 0.5).astype(int)
    
    # Calculate Metrics
    results = []
    
    # XGB Metrics
    results.append({
        "Model": "Model 1 XGBoost", "Split": name,
        "ROC-AUC": roc_auc_score(all_true, xgb_preds),
        "F1": f1_score(all_true, xgb_bin),
        "Precision": precision_score(all_true, xgb_bin),
        "Recall": recall_score(all_true, xgb_bin)
    })
    print(f"XGBoost -> ROC-AUC: {results[-1]['ROC-AUC']:.4f} | F1: {results[-1]['F1']:.4f}")
    
    # LR Metrics
    results.append({
        "Model": "Model 1 Logistic Reg", "Split": name,
        "ROC-AUC": roc_auc_score(all_true, lr_probs),
        "F1": f1_score(all_true, lr_bin),
        "Precision": precision_score(all_true, lr_bin),
        "Recall": recall_score(all_true, lr_bin)
    })
    print(f"Logistic -> ROC-AUC: {results[-1]['ROC-AUC']:.4f} | F1: {results[-1]['F1']:.4f}")
    
    return results

def main():
    args = parse_args()
    
    print("Loading embedding dictionaries into RAM (ChemBERTa, ESM2, BioBERT)...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))
    
    chem_dim = len(next(iter(chem.values())))
    esm_dim = len(next(iter(esm.values())))
    bio_dim = len(next(iter(bio.values())))
    print(f"Dimensions -> Chem: {chem_dim}, ESM: {esm_dim}, Bio: {bio_dim}")
    
    # Init Models
    xgb_params = {"max_depth": 6, "eta": 0.1, "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "n_jobs": -1}
    xgb_model = None
    lr_model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", random_state=42)
    
    chunk_size = 10000
    total_train_lines = sum(1 for _ in open(args.train_csv))
    total_chunks = total_train_lines // chunk_size
    
    print(f"\nTraining Model 1 (XGB & LR) incrementally on {args.train_csv}...")
    is_first_batch = True
    
    for chunk in tqdm(pd.read_csv(args.train_csv, chunksize=chunk_size), total=total_chunks):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, bio, chem_dim, esm_dim, bio_dim)
        
        # Train XGB
        dtrain = xgb.DMatrix(X_batch, label=y_batch)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=10, xgb_model=xgb_model)
        
        # Train LR
        if is_first_batch:
            lr_model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
            is_first_batch = False
        else:
            lr_model.partial_fit(X_batch, y_batch)
            
    print("\nTraining Complete! Running evaluation on S1 and S2...")
    all_results = []
    all_results.extend(evaluate(xgb_model, lr_model, args.s1_csv, chem, esm, bio, chem_dim, esm_dim, bio_dim, "S1 (Known-New)"))
    all_results.extend(evaluate(xgb_model, lr_model, args.s2_csv, chem, esm, bio, chem_dim, esm_dim, bio_dim, "S2 (New-New)"))
    
    # Save
    pd.DataFrame(all_results).to_csv(args.output, index=False)
    print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()