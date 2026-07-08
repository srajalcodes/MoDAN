import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import argparse
from pathlib import Path

# --- Add this right below your imports if it isn't there already! ---
ROOT = Path(__file__).resolve().parents[2]

class FairVanillaMLP(nn.Module):
    def __init__(self, chem_dim=384, esm_dim=1280, bio_dim=768):
        super().__init__()
        input_dim = (chem_dim + esm_dim + bio_dim) * 2 # 4864
    
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)
        )

    def forward(self, drug_a, drug_b):
        x = torch.cat([drug_a, drug_b], dim=-1)
        return self.net(x).squeeze(-1)

class OnTheFlyDDIDataset(Dataset):
    def __init__(self, csv_path, chem_dict, esm_dict, bio_dict, c_dim, e_dim, b_dim):
        self.df = pd.read_csv(csv_path)
        self.chem, self.esm, self.bio = chem_dict, esm_dict, bio_dict
        self.c_dim, self.e_dim, self.b_dim = c_dim, e_dim, b_dim

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1, d2, label = row["drug_A"], row["drug_B"], row["label"]
        c1, e1, b1 = self.chem.get(d1, np.zeros(self.c_dim, dtype=np.float32)), self.esm.get(d1, np.zeros(self.e_dim, dtype=np.float32)), self.bio.get(d1, np.zeros(self.b_dim, dtype=np.float32))
        c2, e2, b2 = self.chem.get(d2, np.zeros(self.c_dim, dtype=np.float32)), self.esm.get(d2, np.zeros(self.e_dim, dtype=np.float32)), self.bio.get(d2, np.zeros(self.b_dim, dtype=np.float32))
        return torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32), torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def evaluate(model, loader, device, name):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for drug_a, drug_b, label in loader:
            logits = model(drug_a.to(device), drug_b.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(label.numpy())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)
    roc = roc_auc_score(all_labels, all_probs)
    print(f"[{name}] ROC-AUC: {roc:.4f}")
    return roc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", default=str(ROOT / "data" / "embeddings" / "chemberta_embeddings.pkl"))
    parser.add_argument("--esm2", default=str(ROOT / "data" / "embeddings" / "esm2_embeddings.pkl"))
    parser.add_argument("--biobert", default=str(ROOT / "data" / "embeddings" / "biobert_drug_embeddings.pkl"))
    parser.add_argument("--train_csv", default=str(ROOT / "data" / "processed" / "train_cold.csv"))
    parser.add_argument("--s1_csv", default=str(ROOT / "data" / "processed" / "test_cold_S1.csv"))
    parser.add_argument("--s2_csv", default=str(ROOT / "data" / "processed" / "test_cold_S2.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chem, esm, bio = pickle.load(open(args.chemberta, "rb")), pickle.load(open(args.esm2, "rb")), pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    train_loader = DataLoader(OnTheFlyDDIDataset(args.train_csv, chem, esm, bio, c_dim, e_dim, b_dim), batch_size=512, shuffle=True)
    s1_loader = DataLoader(OnTheFlyDDIDataset(args.s1_csv, chem, esm, bio, c_dim, e_dim, b_dim), batch_size=512)
    s2_loader = DataLoader(OnTheFlyDDIDataset(args.s2_csv, chem, esm, bio, c_dim, e_dim, b_dim), batch_size=512)

    model = FairVanillaMLP(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_s2 = 0.0
    for epoch in range(1, 6): # 5 epochs is enough to see the ceiling
        model.train()
        for drug_a, drug_b, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss = criterion(model(drug_a.to(device), drug_b.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
            
        roc = evaluate(model, s2_loader, device, f"S2 (Epoch {epoch})")
        if roc > best_s2: best_s2 = roc

    print(f"\n✅ Fair Vanilla MLP Final S2 Score: {best_s2:.4f}")

if __name__ == "__main__":
    main()