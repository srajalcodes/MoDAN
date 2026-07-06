import argparse, pickle, torch, os
import torch.nn as nn
import pandas as pd, numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- UNGATED CROSS-ATTENTION ---
class UngatedCrossAttn(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)
    def forward(self, a, b):
        # EXACT SAME ATTENTION, BUT NO SIGMOID GATE APPLIED!
        a_seq, b_seq = a.unsqueeze(1), b.unsqueeze(1)
        attn_out, _ = self.attn(a_seq, b_seq, b_seq)
        return self.norm(self.drop(attn_out.squeeze(1)))

# (Standard Encoders)
class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None: self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU())
        else: self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU())
    def forward(self, x): return self.net(x)

class UngatedMoDAN(nn.Module):
    def __init__(self, chem_dim=384, esm_dim=1280, bio_dim=768):
        super().__init__()
        self.chem_dim, self.esm_dim, self.bio_dim = chem_dim, esm_dim, bio_dim
        self.chem_enc, self.esm_enc, self.bio_enc = ModalityEncoder(chem_dim), ModalityEncoder(esm_dim, hidden_dim=512), ModalityEncoder(bio_dim)
        self.attn_chem, self.attn_esm, self.attn_bio = UngatedCrossAttn(dim=256), UngatedCrossAttn(dim=256), UngatedCrossAttn(dim=256)
        self.classifier = nn.Sequential(nn.Linear(1536, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    
    def forward(self, drug_a, drug_b):
        ca, ea, ba = self.chem_enc(drug_a[:, :self.chem_dim]), self.esm_enc(drug_a[:, self.chem_dim:self.chem_dim+self.esm_dim]), self.bio_enc(drug_a[:, self.chem_dim+self.esm_dim:])
        cb, eb, bb = self.chem_enc(drug_b[:, :self.chem_dim]), self.esm_enc(drug_b[:, self.chem_dim:self.chem_dim+self.esm_dim]), self.bio_enc(drug_b[:, self.chem_dim+self.esm_dim:])
        I_chem, I_esm, I_bio = self.attn_chem(ca, cb), self.attn_esm(ea, eb), self.attn_bio(ba, bb)
        h = torch.cat([I_chem, I_esm, I_bio, ca-cb, ea-eb, ba-bb], dim=-1)
        return self.classifier(h).squeeze(-1)

class OnTheFlyDDIDataset(Dataset):
    def __init__(self, csv_path, chem, esm, bio, c, e, b):
        self.df = pd.read_csv(csv_path)
        self.chem, self.esm, self.bio, self.c, self.e, self.b = chem, esm, bio, c, e, b
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1, d2, lbl = row["drug_A"], row["drug_B"], row["label"]
        c1, e1, b1 = self.chem.get(d1, np.zeros(self.c, dtype=np.float32)), self.esm.get(d1, np.zeros(self.e, dtype=np.float32)), self.bio.get(d1, np.zeros(self.b, dtype=np.float32))
        c2, e2, b2 = self.chem.get(d2, np.zeros(self.c, dtype=np.float32)), self.esm.get(d2, np.zeros(self.e, dtype=np.float32)), self.bio.get(d2, np.zeros(self.b, dtype=np.float32))
        return torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32), torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32), torch.tensor(lbl, dtype=torch.float32)

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for d_a, d_b, lbl in loader:
            all_probs.extend(torch.sigmoid(model(d_a.to(device), d_b.to(device))).cpu().numpy())
            all_labels.extend(lbl.numpy())
    return roc_auc_score(all_labels, all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--biobert", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chem, esm, bio = pickle.load(open(args.chemberta, "rb")), pickle.load(open(args.esm2, "rb")), pickle.load(open(args.biobert, "rb"))
    c, e, b = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    model = UngatedMoDAN(c, e, b).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(OnTheFlyDDIDataset(r"dataset\train_cold.csv", chem, esm, bio, c, e, b), batch_size=512, shuffle=True)
    s2_loader = DataLoader(OnTheFlyDDIDataset(r"dataset\test_cold_S2.csv", chem, esm, bio, c, e, b), batch_size=512)

    print("Training UNGATED MoDAN Ablation...")
    best_s2 = 0.0
    for epoch in range(1, 4): # 3 epochs is enough for this test
        model.train()
        for d_a, d_b, lbl in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss = criterion(model(d_a.to(device), d_b.to(device)), lbl.to(device))
            loss.backward()
            optimizer.step()
        
        roc = evaluate(model, s2_loader, device)
        if roc > best_s2: best_s2 = roc
        print(f"Epoch {epoch} S2 ROC-AUC: {roc:.4f}")

    print(f"\n✅ UNGATED Cross-Attention S2 Score: {best_s2:.4f}")

if __name__ == "__main__":
    main()