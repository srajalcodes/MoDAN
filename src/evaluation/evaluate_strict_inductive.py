import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------------
# PATH CONFIGURATIONS (Professional pathlib structure)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark_splits"
MODEL_DIR = ROOT / "models"
RESULTS_TABLES_DIR = ROOT / "results" / "tables" / "supplementary"

# Create results folder if it doesn't exist
RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. ARCHITECTURE 
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
            self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU())
    def forward(self, x): return self.net(x)

class ModalAttnDDI(nn.Module):
    def __init__(self, chem_dim=384, esm_dim=1280, bio_dim=768):
        super().__init__()
        self.chem_dim, self.esm_dim, self.bio_dim = chem_dim, esm_dim, bio_dim
        self.chem_enc, self.esm_enc, self.bio_enc = ModalityEncoder(chem_dim), ModalityEncoder(esm_dim, hidden_dim=512), ModalityEncoder(bio_dim)
        self.attn_chem, self.attn_esm, self.attn_bio = GatedCrossAttn(dim=256), GatedCrossAttn(dim=256), GatedCrossAttn(dim=256)
        self.classifier = nn.Sequential(nn.Linear(1536, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def encode(self, x):
        chem = self.chem_enc(x[:, :self.chem_dim])
        esm  = self.esm_enc(x[:, self.chem_dim : self.chem_dim + self.esm_dim])
        bio  = self.bio_enc(x[:, self.chem_dim + self.esm_dim :])
        return chem, esm, bio

    def forward(self, drug_a, drug_b):
        chem_a, esm_a, bio_a = self.encode(drug_a)
        chem_b, esm_b, bio_b = self.encode(drug_b)
        I_chem, I_esm, I_bio = self.attn_chem(chem_a, chem_b), self.attn_esm(esm_a, esm_b), self.attn_bio(bio_a, bio_b)
        D_chem, D_esm, D_bio = chem_a - chem_b, esm_a - esm_b, bio_a - bio_b
        return self.classifier(torch.cat([I_chem, I_esm, I_bio, D_chem, D_esm, D_bio], dim=-1)).squeeze(-1)

class StrictZeroShotDataset(Dataset):
    def __init__(self, df, chem_dict, esm_dict, bio_dict, c_dim, e_dim, b_dim):
        self.df = df
        self.chem, self.esm, self.bio = chem_dict, esm_dict, bio_dict
        self.c_dim, self.e_dim, self.b_dim = c_dim, e_dim, b_dim

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1, d2, label = row["drug_A"], row["drug_B"], row["label"]
        c1, e1, b1 = self.chem.get(d1, np.zeros(self.c_dim, dtype=np.float32)), self.esm.get(d1, np.zeros(self.e_dim, dtype=np.float32)), self.bio.get(d1, np.zeros(self.b_dim, dtype=np.float32))
        c2, e2, b2 = self.chem.get(d2, np.zeros(self.c_dim, dtype=np.float32)), self.esm.get(d2, np.zeros(self.e_dim, dtype=np.float32)), self.bio.get(d2, np.zeros(self.b_dim, dtype=np.float32))
        return torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32), torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# =============================================================================
# 2. EVALUATION FUNCTION
# =============================================================================
def evaluate_strict(name, benchmark_csv, massive_train_csv, model, chem, esm, bio, c_dim, e_dim, b_dim, device):
    # 1. Get seen drugs
    massive_train = pd.read_csv(massive_train_csv)
    seen_drugs = set(massive_train['drug_A']).union(set(massive_train['drug_B']))
    
    # 2. Filter benchmark for TRUE Zero-Shot
    df = pd.read_csv(benchmark_csv)
    total_drugs_in_s2 = set(df['drug_A']).union(set(df['drug_B']))
    leaked_drugs = total_drugs_in_s2.intersection(seen_drugs)
    
    strict_mask = (~df['drug_A'].isin(seen_drugs)) & (~df['drug_B'].isin(seen_drugs))
    strict_df = df[strict_mask].reset_index(drop=True)
    
    print(f"\n--- {name} Strict Zero-Shot ---")
    print(f"Isolated {len(strict_df)} edges where BOTH drugs were never seen during pre-training.")
    
    # Generate stats for Table S02
    overlap_stats = {
        "Dataset": name,
        "Total S2 Drugs": len(total_drugs_in_s2),
        "Overlapping Drugs (Leaked)": len(leaked_drugs),
        "Original S2 Edges": len(df),
        "Strict Zero-Shot Edges (100% Blind)": len(strict_df)
    }

    if len(np.unique(strict_df['label'])) < 2:
        print(f"Skipping {name}: Not enough mixed labels to calculate ROC-AUC.")
        return overlap_stats

    loader = DataLoader(StrictZeroShotDataset(strict_df, chem, esm, bio, c_dim, e_dim, b_dim), batch_size=512, shuffle=False)
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for drug_a, drug_b, label in loader:
            logits = model(drug_a.to(device), drug_b.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(label.numpy())
            
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    
    roc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save Metrics CSVs (Table S09 / S10)
    metric_stats = [{"Metric": "ROC-AUC", "Score": round(roc, 4)}, {"Metric": "F1-Score", "Score": round(f1, 4)}]
    table_num = "s09_biosnap" if name == "BIOSNAP" else "s10_zhangddi"
    pd.DataFrame(metric_stats).to_csv(RESULTS_TABLES_DIR / f"table_{table_num}_strict_zero_shot.csv", index=False)

    return overlap_stats

# =============================================================================
# 3. MAIN RUNNER
# =============================================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # File Paths
    # ------------------------------------------------------------------
    chemberta_path = EMBEDDINGS_DIR / "chemberta_embeddings.pkl"
    esm2_path = EMBEDDINGS_DIR / "esm2_embeddings.pkl"
    biobert_path = EMBEDDINGS_DIR / "biobert_drug_embeddings.pkl"

    # NOTE: Updated to the new, non-confusing model name
    model_path = MODEL_DIR / "modan_final_model.pt"

    train_dataset = PROCESSED_DATA_DIR / "train_cold.csv"
    biosnap_dataset = BENCHMARK_DIR / "BIOSNAP_test_cold_S2.csv"
    zhangddi_dataset = BENCHMARK_DIR / "ZhangDDI_test_cold_S2.csv"

    # ------------------------------------------------------------------
    # Verify Required Files
    # ------------------------------------------------------------------
    required_files = [
        chemberta_path, esm2_path, biobert_path, model_path,
        train_dataset, biosnap_dataset, zhangddi_dataset,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found:\n{file}")

    # ------------------------------------------------------------------
    # Load Embeddings
    # ------------------------------------------------------------------
    print("Loading ChemBERTa embeddings...")
    with open(chemberta_path, "rb") as f: chem = pickle.load(f)

    print("Loading ESM-2 embeddings...")
    with open(esm2_path, "rb") as f: esm = pickle.load(f)

    print("Loading BioBERT embeddings...")
    with open(biobert_path, "rb") as f: bio = pickle.load(f)

    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    # ------------------------------------------------------------------
    # Load Model
    # ------------------------------------------------------------------
    print(f"Loading pretrained MoDAN model from {model_path.name}...")
    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ------------------------------------------------------------------
    # Evaluate & Save Tables
    # ------------------------------------------------------------------
    stat1 = evaluate_strict("BIOSNAP", biosnap_dataset, train_dataset, model, chem, esm, bio, c_dim, e_dim, b_dim, device)
    stat2 = evaluate_strict("ZhangDDI", zhangddi_dataset, train_dataset, model, chem, esm, bio, c_dim, e_dim, b_dim, device)

    # Save the consolidated overlap table (Table S02)
    pd.DataFrame([stat1, stat2]).to_csv(RESULTS_TABLES_DIR / "table_s02_benchmark_overlap_analysis.csv", index=False)

    print("\n✅ Evaluation completed successfully.")
    print(f"✅ Saved Tables S02, S09, and S10 to: {RESULTS_TABLES_DIR}")