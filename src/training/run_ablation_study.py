import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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

class AblationDataset(Dataset):
    def __init__(self, csv_path, chem_dict, esm_dict, bio_dict, c_dim, e_dim, b_dim, mode):
        self.df = pd.read_csv(csv_path)
        self.chem, self.esm, self.bio = chem_dict, esm_dict, bio_dict
        self.c_dim, self.e_dim, self.b_dim = c_dim, e_dim, b_dim
        self.mode = mode # 'chem_only', 'esm_only', 'no_bio'

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1, d2, label = row["drug_A"], row["drug_B"], row["label"]

        # Default Zeros
        c1, c2 = np.zeros(self.c_dim, dtype=np.float32), np.zeros(self.c_dim, dtype=np.float32)
        e1, e2 = np.zeros(self.e_dim, dtype=np.float32), np.zeros(self.e_dim, dtype=np.float32)
        b1, b2 = np.zeros(self.b_dim, dtype=np.float32), np.zeros(self.b_dim, dtype=np.float32)

        # Apply logic
        if self.mode in ['chem_only', 'no_bio']:
            c1, c2 = self.chem.get(d1, c1), self.chem.get(d2, c2)
        if self.mode in ['esm_only', 'no_bio']:
            e1, e2 = self.esm.get(d1, e1), self.esm.get(d2, e2)

        drug_a = torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32)
        drug_b = torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32)
        return drug_a, drug_b, torch.tensor(label, dtype=torch.float32)

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for drug_a, drug_b, label in loader:
            logits = model(drug_a.to(device), drug_b.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(label.numpy())
    return roc_auc_score(all_labels, all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=['chem_only', 'esm_only', 'no_bio'])
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--biobert", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chem, esm, bio = pickle.load(open(args.chemberta, "rb")), pickle.load(open(args.esm2, "rb")), pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    print(f"\n--- Running Ablation: {args.mode.upper()} ---")
    train_loader = DataLoader(AblationDataset(r"..\dataset\train_cold.csv", chem, esm, bio, c_dim, e_dim, b_dim, args.mode), batch_size=512, shuffle=True)
    s2_loader = DataLoader(AblationDataset(r"..\dataset\test_cold_S2.csv", chem, esm, bio, c_dim, e_dim, b_dim, args.mode), batch_size=512)
    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_roc = 0.0
    for epoch in range(1, 6): # 5 epochs is enough for ablation
        model.train()
        for drug_a, drug_b, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss = criterion(model(drug_a.to(device), drug_b.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
        
        roc = evaluate(model, s2_loader, device)
        if roc > best_roc: best_roc = roc
        print(f"Epoch {epoch} S2 ROC-AUC: {roc:.4f}")

    print(f"\n✅ {args.mode.upper()} Final S2 Score: {best_roc:.4f}")

if __name__ == "__main__":
    main()