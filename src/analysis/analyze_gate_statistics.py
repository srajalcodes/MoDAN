import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"

class GatedCrossAttn(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)
        self.last_gate_value = None 
    def forward(self, a, b):
        a_seq, b_seq = a.unsqueeze(1), b.unsqueeze(1)
        attn_out, _ = self.attn(a_seq, b_seq, b_seq)
        attn_out = self.drop(attn_out.squeeze(1))
        g = self.gate(torch.cat([a, b, attn_out], dim=-1))
        
        # Save the mean gate value for this specific batch
        self.last_gate_value = g.mean(dim=-1).detach().cpu().numpy() 
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
        c1, e1, b1 = self.chem.get(d1, np.zeros(self.c_dim, dtype=np.float32)), self.esm.get(d1, np.zeros(self.e_dim, dtype=np.float32)), self.bio.get(d1, np.zeros(self.b_dim, dtype=np.float32))
        c2, e2, b2 = self.chem.get(d2, np.zeros(self.c_dim, dtype=np.float32)), self.esm.get(d2, np.zeros(self.e_dim, dtype=np.float32)), self.bio.get(d2, np.zeros(self.b_dim, dtype=np.float32))
        return torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32), torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", default=str(ROOT / "data" / "embeddings" / "chemberta_embeddings.pkl"))
    parser.add_argument("--esm2", default=str(ROOT / "data" / "embeddings" / "esm2_embeddings.pkl"))
    parser.add_argument("--biobert", default=str(ROOT / "data" / "embeddings" / "biobert_drug_embeddings.pkl"))
    parser.add_argument("--model_path", default=str(ROOT / "models" / "best_biomodal_model.pt"))
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
    model.eval()

    print("\nScanning S2 Unseen dataset for Global Gate Statistics...")
    dataset = OnTheFlyDDIDataset(str(ROOT / "dataset" / "test_cold_S2.csv"), chem, esm, bio, c_dim, e_dim, b_dim)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    chem_gates, esm_gates, bio_gates = [], [], []

    with torch.no_grad():
        for drug_a, drug_b, label in tqdm(loader):
            logits = model(drug_a.to(device), drug_b.to(device))
            
            # Store all gate values for statistical analysis
            chem_gates.extend(model.attn_chem.last_gate_value.tolist())
            esm_gates.extend(model.attn_esm.last_gate_value.tolist())
            bio_gates.extend(model.attn_bio.last_gate_value.tolist())
            
    chem_gates = np.array(chem_gates)
    esm_gates = np.array(esm_gates)
    bio_gates = np.array(bio_gates)

    print("\n" + "="*50)
    print("GLOBAL STATISTICAL GATE ANALYSIS (N=100,694)")
    print("="*50)
    print(f"Chemistry Gate: {np.mean(chem_gates):.4f} ± {np.std(chem_gates):.4f}")
    print(f"Protein Gate:   {np.mean(esm_gates):.4f} ± {np.std(esm_gates):.4f}")
    print(f"Pathway Gate:   {np.mean(bio_gates):.4f} ± {np.std(bio_gates):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()