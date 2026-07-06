import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from tqdm import tqdm

# =============================================================================
# 1. ARCHITECTURE (Must match the saved model exactly)
# =============================================================================
class GatedCrossAttn(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, a, b):
        a_seq, b_seq = a.unsqueeze(1), b.unsqueeze(1)
        attn_out, _ = self.attn(a_seq, b_seq, b_seq)
        attn_out = self.drop(attn_out.squeeze(1))
        g = self.gate(torch.cat([a, b, attn_out], dim=-1))
        return self.norm(attn_out * g)

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU())
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU()
            )
    def forward(self, x): return self.net(x)

class ModalAttnDDI(nn.Module):
    def __init__(self, chem_dim=384, esm_dim=1280, bio_dim=768):
        super().__init__()
        self.chem_dim, self.esm_dim, self.bio_dim = chem_dim, esm_dim, bio_dim
        self.chem_enc = ModalityEncoder(chem_dim)
        self.esm_enc  = ModalityEncoder(esm_dim, hidden_dim=512)
        self.bio_enc  = ModalityEncoder(bio_dim)
        self.attn_chem = GatedCrossAttn(dim=256)
        self.attn_esm  = GatedCrossAttn(dim=256)
        self.attn_bio  = GatedCrossAttn(dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def encode(self, x):
        chem = self.chem_enc(x[:, :self.chem_dim])
        esm  = self.esm_enc(x[:, self.chem_dim : self.chem_dim + self.esm_dim])
        bio  = self.bio_enc(x[:, self.chem_dim + self.esm_dim :])
        return chem, esm, bio

    def forward(self, drug_a, drug_b):
        chem_a, esm_a, bio_a = self.encode(drug_a)
        chem_b, esm_b, bio_b = self.encode(drug_b)
        I_chem = self.attn_chem(chem_a, chem_b)
        I_esm  = self.attn_esm(esm_a, esm_b)
        I_bio  = self.attn_bio(bio_a, bio_b)
        D_chem, D_esm, D_bio = chem_a - chem_b, esm_a - esm_b, bio_a - bio_b
        h = torch.cat([I_chem, I_esm, I_bio, D_chem, D_esm, D_bio], dim=-1)
        return self.classifier(h).squeeze(-1)

class OnTheFlyDDIDataset(Dataset):
    def __init__(self, csv_path, chem_dict, esm_dict, bio_dict, c_dim, e_dim, b_dim):
        self.df = pd.read_csv(csv_path)
        self.chem, self.esm, self.bio = chem_dict, esm_dict, bio_dict
        self.c_dim, self.e_dim, self.b_dim = c_dim, e_dim, b_dim

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1, d2, label = row["drug_A"], row["drug_B"], row["label"]
        c1 = self.chem.get(d1, np.zeros(self.c_dim, dtype=np.float32))
        e1 = self.esm.get(d1, np.zeros(self.e_dim, dtype=np.float32))
        b1 = self.bio.get(d1, np.zeros(self.b_dim, dtype=np.float32))
        c2 = self.chem.get(d2, np.zeros(self.c_dim, dtype=np.float32))
        e2 = self.esm.get(d2, np.zeros(self.e_dim, dtype=np.float32))
        b2 = self.bio.get(d2, np.zeros(self.b_dim, dtype=np.float32))
        drug_a = torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32)
        drug_b = torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32)
        return drug_a, drug_b, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 3. BOOTSTRAP STATISTICAL TESTING
# =============================================================================
def bootstrap_metrics(y_true, y_prob, n_bootstraps=1000):
    """ Resamples predictions 1000 times to calculate standard deviation and PR-AUC """
    print("Running 1000-iteration bootstrap for statistical significance...")
    np.random.seed(42)
    metrics = {'ROC-AUC': [], 'PR-AUC': [], 'F1': [], 'Precision': [], 'Recall': []}
    
    for _ in tqdm(range(n_bootstraps), leave=False):
        # Sample with replacement
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: 
            continue # Skip invalid splits
            
        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]
        y_pred_b = (y_prob_b >= 0.5).astype(int)
        
        metrics['ROC-AUC'].append(roc_auc_score(y_true_b, y_prob_b))
        metrics['PR-AUC'].append(average_precision_score(y_true_b, y_prob_b)) # NEW METRIC!
        metrics['F1'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
        metrics['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
        metrics['Recall'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
        
    summary = {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in metrics.items()}
    return summary

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
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    results = bootstrap_metrics(all_labels, all_probs)
    print(f"[{name}] Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--biobert", required=True)
    parser.add_argument("--model_path", default=r"C:\Users\st735\OneDrive - Shiv Nadar Institution of Eminence\Documents\CODE\Drug Drug Interaction\FInal_model\best_biomodal_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading dictionaries...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    print("Loading Pre-Trained Model...")
    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 1. Custom Dataset S1 & S2
    evaluate_and_bootstrap(model, r"C:\Users\st735\OneDrive - Shiv Nadar Institution of Eminence\Documents\CODE\Drug Drug Interaction\dataset\test_cold_S1.csv", chem, esm, bio, c_dim, e_dim, b_dim, device, "DrugBank Custom S1")
    evaluate_and_bootstrap(model, r"C:\Users\st735\OneDrive - Shiv Nadar Institution of Eminence\Documents\CODE\Drug Drug Interaction\dataset\test_cold_S2.csv", chem, esm, bio, c_dim, e_dim, b_dim, device, "DrugBank Custom S2")
    
    # 2. Benchmarks S2
    evaluate_and_bootstrap(model, r"C:\Users\st735\OneDrive - Shiv Nadar Institution of Eminence\Documents\CODE\Drug Drug Interaction\benchmark_splits\BIOSNAP_test_cold_S2.csv", chem, esm, bio, c_dim, e_dim, b_dim, device, "BIOSNAP S2 Transfer")
    evaluate_and_bootstrap(model, r"C:\Users\st735\OneDrive - Shiv Nadar Institution of Eminence\Documents\CODE\Drug Drug Interaction\benchmark_splits\ZhangDDI_test_cold_S2.csv", chem, esm, bio, c_dim, e_dim, b_dim, device, "ZhangDDI S2 Transfer")

if __name__ == "__main__":
    main()