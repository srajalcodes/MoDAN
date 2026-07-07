import argparse
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from tqdm import tqdm
import os

# Fix for VS Code OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_features_for_batch(df_chunk, chem, esm, bio, c_dim, e_dim, b_dim):
    X, y = [], []
    for _, row in df_chunk.iterrows():
        d1, d2, label = row["drug_A"], row["drug_B"], row["label"]
        c1 = chem.get(d1, np.zeros(c_dim, dtype=np.float32))
        e1 = esm.get(d1, np.zeros(e_dim, dtype=np.float32))
        b1 = bio.get(d1, np.zeros(b_dim, dtype=np.float32))
        c2 = chem.get(d2, np.zeros(c_dim, dtype=np.float32))
        e2 = esm.get(d2, np.zeros(e_dim, dtype=np.float32))
        b2 = bio.get(d2, np.zeros(b_dim, dtype=np.float32))
        
        vec1 = np.concatenate([c1, e1, b1])
        vec2 = np.concatenate([c2, e2, b2])
        X.append(np.concatenate([vec1, vec2]))
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def bootstrap_metrics(y_true, y_prob, n_bootstraps=1000):
    np.random.seed(42)
    metrics = {'ROC-AUC': [], 'PR-AUC': [], 'F1': [], 'Precision': [], 'Recall': []}
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue 
        y_true_b, y_prob_b = y_true[indices], y_prob[indices]
        y_pred_b = (y_prob_b >= 0.5).astype(int)
        
        metrics['ROC-AUC'].append(roc_auc_score(y_true_b, y_prob_b))
        metrics['PR-AUC'].append(average_precision_score(y_true_b, y_prob_b))
        metrics['F1'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
        metrics['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
        metrics['Recall'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
        
    return {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()}

def evaluate_models(xgb_model, lr_model, csv_path, chem, esm, bio, c_dim, e_dim, b_dim, name):
    print(f"\nEvaluating {name}...")
    xgb_probs, lr_probs, all_true = [], [], []
    
    # Read in chunks to avoid memory crash
    for chunk in pd.read_csv(csv_path, chunksize=10000):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, bio, c_dim, e_dim, b_dim)
        dtest = xgb.DMatrix(X_batch)
        xgb_probs.extend(xgb_model.predict(dtest))
        lr_probs.extend(lr_model.predict_proba(X_batch)[:, 1])
        all_true.extend(y_batch)
        
    xgb_probs, lr_probs, all_true = np.array(xgb_probs), np.array(lr_probs), np.array(all_true)
    
    print("  Bootstrapping XGBoost...")
    xgb_res = bootstrap_metrics(all_true, xgb_probs)
    for k, v in xgb_res.items(): print(f"    XGB {k}: {v}")
        
    print("  Bootstrapping Logistic Regression...")
    lr_res = bootstrap_metrics(all_true, lr_probs)
    for k, v in lr_res.items(): print(f"    LR {k}: {v}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--biobert", required=True)
    args = parser.parse_args()

    print("Loading dictionaries...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    # Init Models
    xgb_params = {"max_depth": 6, "eta": 0.1, "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "n_jobs": -1, "seed": 42}
    xgb_model = None
    lr_model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", random_state=42)

    print("\nTraining Baselines (XGBoost & LR) incrementally...")
    total_chunks = sum(1 for _ in open(PROCESSED_DATA_DIR / "train_cold.csv")) // 10000
    is_first = True
    
    for chunk in tqdm(pd.read_csv(PROCESSED_DATA_DIR / "train_cold.csv", chunksize=10000), total=total_chunks):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, bio, c_dim, e_dim, b_dim)
        
        xgb_model = xgb.train(xgb_params, xgb.DMatrix(X_batch, label=y_batch), num_boost_round=10, xgb_model=xgb_model)
        if is_first:
            lr_model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
            is_first = False
        else:
            lr_model.partial_fit(X_batch, y_batch)

    # Evaluate and Bootstrap
    evaluate_models(xgb_model, lr_model, PROCESSED_DATA_DIR / "test_cold_S1.csv", chem, esm, bio, c_dim, e_dim, b_dim, "S1 (Known-New)")
    evaluate_models(xgb_model, lr_model, PROCESSED_DATA_DIR / "test_cold_S2.csv", chem, esm, bio, c_dim, e_dim, b_dim, "S2 (New-New)")

if __name__ == "__main__":
    main()