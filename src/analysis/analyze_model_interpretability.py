import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

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
        self.last_gate_value = None 

    def forward(self, a, b):
        a_seq, b_seq = a.unsqueeze(1), b.unsqueeze(1)
        attn_out, _ = self.attn(a_seq, b_seq, b_seq)
        attn_out = self.drop(attn_out.squeeze(1))
        
        g = self.gate(torch.cat([a, b, attn_out], dim=-1))
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

# =============================================================================
# 2. EVALUATION & EXTRACTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--biobert", required=True)
    parser.add_argument("--drug_meta", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    meta_df = pd.read_csv(args.drug_meta)
    id_to_name = dict(zip(meta_df['drugbank_id'], meta_df['name']))

    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    s2_df = pd.read_csv(args.test_csv)
    
    print("\nScanning S2 Unseen dataset for highly confident predictions...")
    results = []

    with torch.no_grad():
        for idx, row in s2_df.iterrows():
            d1, d2, label = row["drug_A"], row["drug_B"], row["label"]
            
            if label != 1:
                continue

            c1, e1, b1 = chem.get(d1, np.zeros(c_dim, dtype=np.float32)), esm.get(d1, np.zeros(e_dim, dtype=np.float32)), bio.get(d1, np.zeros(b_dim, dtype=np.float32))
            c2, e2, b2 = chem.get(d2, np.zeros(c_dim, dtype=np.float32)), esm.get(d2, np.zeros(e_dim, dtype=np.float32)), bio.get(d2, np.zeros(b_dim, dtype=np.float32))
            
            drug_a = torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32).unsqueeze(0).to(device)
            drug_b = torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32).unsqueeze(0).to(device)
            
            logits = model(drug_a, drug_b)
            prob = torch.sigmoid(logits).item()
            
            if prob > 0.95:
                results.append({
                    "Drug_1": id_to_name.get(d1, d1),
                    "Drug_2": id_to_name.get(d2, d2),
                    "Probability": prob,
                    "Gate_Chem": model.attn_chem.last_gate_value[0],
                    "Gate_Protein": model.attn_esm.last_gate_value[0],
                    "Gate_Pathway": model.attn_bio.last_gate_value[0]
                })
                
            if len(results) >= 5:
                break

    print("\n" + "="*60)
    print("🔬 INTRINSIC EXPLAINABILITY (CASE STUDIES) 🔬")
    print("="*60)
    for i, res in enumerate(results):
        print(f"\nPair {i+1}: {res['Drug_1']} + {res['Drug_2']}")
        print(f"Prediction Confidence: {res['Probability']*100:.1f}%")
        print(f"  -> Chemistry Gate Activation: {res['Gate_Chem']:.3f}")
        print(f"  -> Protein Gate Activation:   {res['Gate_Protein']:.3f}")
        print(f"  -> Pathway Gate Activation:   {res['Gate_Pathway']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()