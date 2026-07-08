import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from tqdm import tqdm

# --- Fix for VS Code OpenMP crash ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Add Repository Root to Python Path ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# --- Import Core Architecture ---
from src.model import ModalAttnDDI
from src.dataset import OnTheFlyDDIDataset

# =============================================================================
# 1. BOOTSTRAP STATISTICAL TESTING
# =============================================================================
def bootstrap_metrics(y_true, y_prob, n_bootstraps=1000):
    """ Resamples predictions 1000 times to calculate standard deviation and PR-AUC """
    print("Running 1000-iteration bootstrap for statistical significance...")
    np.random.seed(42)
    metrics = {'ROC-AUC': [], 'PR-AUC': [], 'F1': [], 'Precision': [], 'Recall': []}
    
    for _ in tqdm(range(n_bootstraps), leave=False):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: 
            continue # Skip invalid splits
            
        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]
        y_pred_b = (y_prob_b >= 0.5).astype(int)
        
        metrics['ROC-AUC'].append(roc_auc_score(y_true_b, y_prob_b))
        metrics['PR-AUC'].append(average_precision_score(y_true_b, y_prob_b))
        metrics['F1'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
        metrics['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
        metrics['Recall'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
        
    return {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()}

def evaluate_and_bootstrap(model, csv_path, chem, esm, bio, c_dim, e_dim, b_dim, device, name):
    print(f"\nEvaluating {name}...")
    dataset = OnTheFlyDDIDataset(csv_path, chem, esm, bio, c_dim, e_dim, b_dim)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for drug_a, drug_b, label in loader:
            logits = model(drug_a.to(device), drug_b.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(label.numpy())
            
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    results = bootstrap_metrics(all_labels, all_probs)
    
    print(f"[{name}] Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
        
    return {
        "Model": "MoDAN", "Split": name,
        "ROC-AUC": results["ROC-AUC"], "PR-AUC": results["PR-AUC"],
        "F1-Score": results["F1"], "Precision": results["Precision"], "Recall": results["Recall"]
    }

def main():
    parser = argparse.ArgumentParser()
    # Dynamic Relative Paths
    parser.add_argument("--chemberta", default=str(ROOT / "data" / "embeddings" / "chemberta_embeddings.pkl"))
    parser.add_argument("--esm2", default=str(ROOT / "data" / "embeddings" / "esm2_embeddings.pkl"))
    parser.add_argument("--biobert", default=str(ROOT / "data" / "embeddings" / "biobert_drug_embeddings.pkl"))
    parser.add_argument("--model_path", default=str(ROOT / "models" / "modan_final_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    print("Loading dictionaries...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    print("Loading Pre-Trained Model...")
    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    results_list = []
    
    # 1. Custom Dataset S1 & S2
    results_list.append(evaluate_and_bootstrap(model, str(ROOT / "data" / "processed" / "test_cold_S1.csv"), chem, esm, bio, c_dim, e_dim, b_dim, device, "DrugBank Custom S1"))
    results_list.append(evaluate_and_bootstrap(model, str(ROOT / "data" / "processed" / "test_cold_S2.csv"), chem, esm, bio, c_dim, e_dim, b_dim, device, "DrugBank Custom S2"))
    
    # 2. Benchmarks S2
    results_list.append(evaluate_and_bootstrap(model, str(ROOT / "data" / "benchmark_splits" / "BIOSNAP_test_cold_S2.csv"), chem, esm, bio, c_dim, e_dim, b_dim, device, "BIOSNAP S2 Transfer"))
    results_list.append(evaluate_and_bootstrap(model, str(ROOT / "data" / "benchmark_splits" / "ZhangDDI_test_cold_S2.csv"), chem, esm, bio, c_dim, e_dim, b_dim, device, "ZhangDDI S2 Transfer"))

    # Save to CSV
    os.makedirs(str(ROOT / "results" / "tables" / "main"), exist_ok=True)
    df = pd.DataFrame(results_list)
    out_path = str(ROOT / "results" / "tables" / "main" / "modan_final_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved results to {out_path}")

if __name__ == "__main__":
    main()