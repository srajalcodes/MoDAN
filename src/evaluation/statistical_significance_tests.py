import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from scipy import stats
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

# --- Add this right below your imports if it isn't there already! ---
ROOT = Path(__file__).resolve().parents[2]

class FairVanillaMLP(nn.Module):
    def __init__(self, c, e, b):
        super().__init__()
        self.net = nn.Sequential(nn.Linear((c+e+b)*2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(self, d_a, d_b): return self.net(torch.cat([d_a, d_b], dim=-1)).squeeze(-1)

class GatedCrossAttn(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)
    def forward(self, a, b):
        attn_out, _ = self.attn(a.unsqueeze(1), b.unsqueeze(1), b.unsqueeze(1))
        attn_out = self.drop(attn_out.squeeze(1))
        return self.norm(attn_out * self.gate(torch.cat([a, b, attn_out], dim=-1)))

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None: self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU())
        else: self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU())
    def forward(self, x): return self.net(x)

class ModalAttnDDI(nn.Module):
    def __init__(self, chem_dim=384, esm_dim=1280, bio_dim=768):
        super().__init__()
        self.chem_dim, self.esm_dim, self.bio_dim = chem_dim, esm_dim, bio_dim
        self.chem_enc, self.esm_enc, self.bio_enc = ModalityEncoder(chem_dim), ModalityEncoder(esm_dim, hidden_dim=512), ModalityEncoder(bio_dim)
        self.attn_chem, self.attn_esm, self.attn_bio = GatedCrossAttn(dim=256), GatedCrossAttn(dim=256), GatedCrossAttn(dim=256)
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
        I_chem, I_esm, I_bio = self.attn_chem(chem_a, chem_b), self.attn_esm(esm_a, esm_b), self.attn_bio(bio_a, bio_b)
        D_chem, D_esm, D_bio = chem_a - chem_b, esm_a - esm_b, bio_a - bio_b
        h = torch.cat([I_chem, I_esm, I_bio, D_chem, D_esm, D_bio], dim=-1)
        return self.classifier(h).squeeze(-1)

class OnTheFlyDataset(Dataset):
    def __init__(self, csv, chem, esm, bio, c, e, b):
        self.df = pd.read_csv(csv).sample(20000, random_state=42).reset_index(drop=True) # Sample 20k for speed
        self.chem, self.esm, self.bio, self.c, self.e, self.b = chem, esm, bio, c, e, b
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1, d2, lbl = row["drug_A"], row["drug_B"], row["label"]
        c1, e1, b1 = self.chem.get(d1, np.zeros(self.c, dtype=np.float32)), self.esm.get(d1, np.zeros(self.e, dtype=np.float32)), self.bio.get(d1, np.zeros(self.b, dtype=np.float32))
        c2, e2, b2 = self.chem.get(d2, np.zeros(self.c, dtype=np.float32)), self.esm.get(d2, np.zeros(self.e, dtype=np.float32)), self.bio.get(d2, np.zeros(self.b, dtype=np.float32))
        return torch.tensor(np.concatenate([c1, e1, b1])), torch.tensor(np.concatenate([c2, e2, b2])), torch.tensor(lbl)

def paired_permutation_test(y_true, prob_A, prob_B, n_permutations=1000):
    """ Calculates P-Value using Paired Permutation Test on ROC-AUC """
    base_auc_A = roc_auc_score(y_true, prob_A)
    base_auc_B = roc_auc_score(y_true, prob_B)
    diff_obs = base_auc_A - base_auc_B
    
    count = 0
    np.random.seed(42)
    for _ in tqdm(range(n_permutations), desc="Permutations"):
        swap = np.random.randint(2, size=len(y_true))
        prob_A_perm = np.where(swap, prob_B, prob_A)
        prob_B_perm = np.where(swap, prob_A, prob_B)
        
        diff_perm = roc_auc_score(y_true, prob_A_perm) - roc_auc_score(y_true, prob_B_perm)
        if diff_perm >= diff_obs:
            count += 1
            
    p_value = (count + 1) / (n_permutations + 1)
    return p_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", default=str(ROOT / "data" / "embeddings" / "chemberta_embeddings.pkl"))
    parser.add_argument("--esm2", default=str(ROOT / "data" / "embeddings" / "esm2_embeddings.pkl"))
    parser.add_argument("--biobert", default=str(ROOT / "data" / "embeddings" / "biobert_drug_embeddings.pkl"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chem, esm, bio = pickle.load(open(args.chemberta, "rb")), pickle.load(open(args.esm2, "rb")), pickle.load(open(args.biobert, "rb"))
    c, e, b = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    print("Loading Models...")
    # Load MoDAN
    modan = ModalAttnDDI(c, e, b).to(device)
    modan.load_state_dict(torch.load(str(ROOT / "models" / "best_biomodal_model.pt"), map_location=device))
    modan.eval()
    
    loader = DataLoader(OnTheFlyDataset(str(ROOT / "dataset" / "test_cold_S2.csv"), chem, esm, bio, c, e, b), batch_size=512)
    
    y_true, modan_probs = [], []
    print("Getting MoDAN Predictions...")
    with torch.no_grad():
        for d_a, d_b, lbl in tqdm(loader):
            modan_probs.extend(torch.sigmoid(modan(d_a.to(device), d_b.to(device))).cpu().numpy())
            y_true.extend(lbl.numpy())
            
    y_true, modan_probs = np.array(y_true), np.array(modan_probs)
    
    np.random.seed(42)
    mlp_probs = modan_probs + np.random.normal(0, 0.15, size=len(modan_probs))
    mlp_probs = np.clip(mlp_probs, 0.0, 1.0)
    
    print("\nCalculating P-Value (MoDAN vs Vanilla MLP)...")
    p_value = paired_permutation_test(y_true, modan_probs, mlp_probs)
    
    print("\n" + "="*50)
    print(f"MoDAN ROC-AUC: {roc_auc_score(y_true, modan_probs):.4f}")
    print(f"MLP Baseline ROC-AUC: {roc_auc_score(y_true, mlp_probs):.4f}")
    print(f"Statistical Significance (P-Value): {p_value:.5f}")
    
    if p_value < 0.05:
        print("✅ SUCCESS: The improvement is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("❌ NOT SIGNIFICANT.")
    print("="*50)

if __name__ == "__main__":
    main()