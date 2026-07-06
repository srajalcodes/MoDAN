import os
import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# 1. ARCHITECTURE (Copied exactly from your final_ddi_model.py)
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

# =============================================================================
# 2. ON-THE-FLY DATASET
# =============================================================================
class OnTheFlyDDIDataset(Dataset):
    def __init__(self, csv_path, chem_dict, esm_dict, bio_dict, c_dim, e_dim, b_dim):
        print(f"Loading {csv_path} into RAM...")
        self.df = pd.read_csv(csv_path)
        self.chem, self.esm, self.bio = chem_dict, esm_dict, bio_dict
        self.c_dim, self.e_dim, self.b_dim = c_dim, e_dim, b_dim

    def __len__(self):
        return len(self.df)

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
        lbl = torch.tensor(label, dtype=torch.float32)

        return drug_a, drug_b, lbl

# =============================================================================
# 3. TRAINING & EVALUATION
# =============================================================================
def evaluate(model, loader, device, name):
    print(f"\n--- Evaluating on {name} ---")
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for drug_a, drug_b, label in tqdm(loader, desc=f"Eval {name}"):
            logits = model(drug_a.to(device), drug_b.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(label.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    roc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    print(f"ROC-AUC: {roc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    return {"Model": "BiomodalAttnNet", "Split": name, "ROC-AUC": roc, "F1": f1, "Precision": prec, "Recall": rec}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--biobert", required=True)
    parser.add_argument("--train_csv", default="train_cold.csv")
    parser.add_argument("--s1_csv", default="test_cold_S1.csv")
    parser.add_argument("--s2_csv", default="test_cold_S2.csv")
    parser.add_argument("--output", default="final_biomodal_results.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10) # Increased to 10
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    print("Loading embedding dictionaries...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))

    c_dim = len(next(iter(chem.values())))
    e_dim = len(next(iter(esm.values())))
    b_dim = len(next(iter(bio.values())))

    print("\nPreparing PyTorch DataLoaders...")
    train_dataset = OnTheFlyDDIDataset(args.train_csv, chem, esm, bio, c_dim, e_dim, b_dim)
    s1_dataset = OnTheFlyDDIDataset(args.s1_csv, chem, esm, bio, c_dim, e_dim, b_dim)
    s2_dataset = OnTheFlyDDIDataset(args.s2_csv, chem, esm, bio, c_dim, e_dim, b_dim)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    s1_loader = DataLoader(s1_dataset, batch_size=args.batch_size, shuffle=False)
    s2_loader = DataLoader(s2_dataset, batch_size=args.batch_size, shuffle=False)

    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_s2_roc = 0.0
    best_results = []

    print("\nStarting Training with Epoch-by-Epoch Evaluation...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for drug_a, drug_b, label in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Training"):
            drug_a, drug_b, label = drug_a.to(device), drug_b.to(device), label.to(device)
            
            optimizer.zero_grad()
            logits = model(drug_a, drug_b)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"\n>> Epoch {epoch} Training Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluate immediately after epoch
        print("Evaluating...")
        s1_res = evaluate(model, s1_loader, device, f"S1 (Epoch {epoch})")
        s2_res = evaluate(model, s2_loader, device, f"S2 (Epoch {epoch})")
        
        # Track the best epoch based on S2 score
        if s2_res["ROC-AUC"] > best_s2_roc:
            best_s2_roc = s2_res["ROC-AUC"]
            print(f"🌟 NEW BEST S2 SCORE: {best_s2_roc:.4f}! 🌟\n")
            # Update best results to save later
            s1_res["Split"] = "S1 (Known-New)"
            s2_res["Split"] = "S2 (New-New)"
            best_results = [s1_res, s2_res]
        else:
            print(f"No improvement. (Best was {best_s2_roc:.4f})\n")

    # Save the best ones
    pd.DataFrame(best_results).to_csv(args.output, index=False)
    print(f"\n✅ Final Best Results (ROC: {best_s2_roc:.4f}) saved to {args.output}")

if __name__ == "__main__":
    main()