import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--train_csv", default="train_cold.csv")
    parser.add_argument("--s1_csv", default="test_cold_S1.csv")
    parser.add_argument("--s2_csv", default="test_cold_S2.csv")
    parser.add_argument("--output", default="lr_cold_results.csv")
    return parser.parse_args()

def get_features_for_batch(df_chunk, chem, esm, chem_dim, esm_dim):
    X, y = [], []
    for _, row in df_chunk.iterrows():
        d1, d2 = row["drug_A"], row["drug_B"]
        vec1 = np.concatenate([chem.get(d1, np.zeros(chem_dim)), esm.get(d1, np.zeros(esm_dim))])
        vec2 = np.concatenate([chem.get(d2, np.zeros(chem_dim)), esm.get(d2, np.zeros(esm_dim))])
        X.append(np.concatenate([vec1, vec2]))
        y.append(row["label"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def evaluate(model, csv_path, chem, esm, chem_dim, esm_dim, name):
    print(f"\n--- Evaluating on {name} ---")
    all_probs = []
    all_true = []
    chunk_size = 10000
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, chem_dim, esm_dim)
        probs = model.predict_proba(X_batch)[:, 1] # Get probability of class 1
        all_probs.extend(probs)
        all_true.extend(y_batch)
        
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    bin_preds = (all_probs > 0.5).astype(int)
    
    roc = roc_auc_score(all_true, all_probs)
    f1 = f1_score(all_true, bin_preds)
    precision = precision_score(all_true, bin_preds)
    recall = recall_score(all_true, bin_preds)
    
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    return {"Split": name, "ROC-AUC": roc, "F1": f1, "Precision": precision, "Recall": recall}

def main():
    args = parse_args()
    
    print("Loading embedding dictionaries into RAM...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    chem_dim = len(next(iter(chem.values())))
    esm_dim = len(next(iter(esm.values())))
    
    # Initialize Logistic Regression (SGD)
    model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", random_state=42)
    
    chunk_size = 10000
    total_train_lines = sum(1 for _ in open(args.train_csv))
    total_chunks = total_train_lines // chunk_size
    
    print(f"\nTraining Logistic Regression incrementally on {args.train_csv}...")
    is_first_batch = True
    
    for chunk in tqdm(pd.read_csv(args.train_csv, chunksize=chunk_size), total=total_chunks):
        X_batch, y_batch = get_features_for_batch(chunk, chem, esm, chem_dim, esm_dim)
        
        # SGDClassifier requires knowing the classes [0, 1] on the very first batch
        if is_first_batch:
            model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
            is_first_batch = False
        else:
            model.partial_fit(X_batch, y_batch)
            
    print("\nTraining Complete! Now running evaluation...")
    results = []
    results.append(evaluate(model, args.s1_csv, chem, esm, chem_dim, esm_dim, "S1 (Known-New)"))
    results.append(evaluate(model, args.s2_csv, chem, esm, chem_dim, esm_dim, "S2 (New-New)"))
    
    # SAVE TO CSV
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"\n✅ Results successfully saved to {args.output}")

if __name__ == "__main__":
    main()