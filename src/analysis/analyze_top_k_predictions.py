import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"

BIOLOGICAL_DIR = DATA_DIR / "biological"

METADATA_DIR = DATA_DIR / "metadata"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

BENCHMARK_DIR = DATA_DIR / "benchmark_splits"

MODEL_DIR = ROOT / "models"

RESULTS_DIR = ROOT / "results"

CONFIG_DIR = ROOT / "configs"

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    print("Loading embedding dictionaries...")
    chem = pickle.load(open(EMBEDDINGS_DIR / "chemberta_embeddings.pkl", "rb"))
    esm = pickle.load(open(EMBEDDINGS_DIR / "esm2_embeddings.pkl", "rb"))
    bio = pickle.load(open(EMBEDDINGS_DIR / "biobert_drug_embeddings.pkl", "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    print("Loading Drug Metadata...")
    meta_df = pd.read_csv(METADATA_DIR / "final_drug_nodes.csv")
    id_to_name = dict(zip(meta_df['drugbank_id'], meta_df['name']))
    all_drug_ids = list(meta_df['drugbank_id'].unique())

    print("Loading Pre-Trained MoDAN Model...")
    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best_biomodal_model.pt", map_location=device))
    model.eval()

    # The Query Drug: RITONAVIR (DB00503)
    query_id = "DB00503"
    query_name = id_to_name.get(query_id, "Ritonavir")
    
    c1 = chem.get(query_id, np.zeros(c_dim, dtype=np.float32))
    e1 = esm.get(query_id, np.zeros(e_dim, dtype=np.float32))
    b1 = bio.get(query_id, np.zeros(b_dim, dtype=np.float32))
    drug_a = torch.tensor(np.concatenate([c1, e1, b1]), dtype=torch.float32).unsqueeze(0).to(device)

    print(f"\n🚀 Initiating High-Throughput Screening for {query_name} ({query_id}) against {len(all_drug_ids)} drugs...")
    
    results = []
    with torch.no_grad():
        for target_id in tqdm(all_drug_ids, desc="Scanning Database"):
            if target_id == query_id:
                continue 
                
            c2 = chem.get(target_id, np.zeros(c_dim, dtype=np.float32))
            e2 = esm.get(target_id, np.zeros(e_dim, dtype=np.float32))
            b2 = bio.get(target_id, np.zeros(b_dim, dtype=np.float32))
            drug_b = torch.tensor(np.concatenate([c2, e2, b2]), dtype=torch.float32).unsqueeze(0).to(device)

            logits = model(drug_a, drug_b)
            prob = torch.sigmoid(logits).item()
            
            results.append({
                "Target Name": id_to_name.get(target_id, target_id),
                "Probability": prob,
                "Chemistry Gate": model.attn_chem.last_gate_value[0],
                "Protein Gate": model.attn_esm.last_gate_value[0],
                "Pathway Gate": model.attn_bio.last_gate_value[0]
            })

    df_results = pd.DataFrame(results).sort_values(by="Probability", ascending=False)
    

    top_5 = df_results.head(5).copy()
    bottom_5 = df_results.tail(5).copy()
    combined_df = pd.concat([top_5, bottom_5])

    combined_df["Label"] = combined_df.apply(
        lambda row: f"{query_name} + {row['Target Name']} ({row['Probability']*100:.1f}%)", axis=1
    )
    combined_df.set_index("Label", inplace=True)
    gate_data = combined_df[["Chemistry Gate", "Protein Gate", "Pathway Gate"]]

    combined_df = pd.concat([top_5, bottom_5])

    combined_df["Label"] = combined_df.apply(
        lambda row: f"{query_name} + {row['Target Name']} ({row['Probability']*100:.1f}%)", axis=1
    )
    combined_df.set_index("Label", inplace=True)
    gate_data = combined_df[["Chemistry Gate", "Protein Gate", "Pathway Gate"]]

    csv_filename = RESULTS_DIR / "top_k_discovery_results.csv"
    gate_data.to_csv(csv_filename)
    print(f"\n💾 Saved raw results to {csv_filename}!")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(gate_data, annot=True, fmt=".3f", cmap="vlag", 
                linewidths=2, linecolor='white', ax=ax)
    
    ax.axhline(5, color='black', linewidth=3, linestyle='')

    cbar = ax.collections[0].colorbar
    cbar.set_label('Gate Activation (g)', fontweight='bold', fontsize=12)


    ax.text(-3.5, 2.5, '', color='darkred', weight='bold', rotation=90, va='center', ha='center', fontsize=13)
    ax.text(-3.5, 7.5, '', color='darkblue', weight='bold', rotation=90, va='center', ha='center', fontsize=13)

    plt.title(f"High-Throughput Discovery & Negative Controls: {query_name}", pad=20, weight='bold', fontsize=15)
    plt.ylabel("") 
    
    plt.xticks(weight='bold', fontsize=12, rotation=45, ha='right')
    plt.yticks(weight='bold', fontsize=11, rotation=0)
    
    plt.tight_layout()
    
    plt.savefig(RESULTS_DIR / "Figure_6_TopK_Discovery.pdf", dpi=1200, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / "Figure_6_TopK_Discovery.png", dpi=1200, bbox_inches='tight')
    
    print("✅ High-Resolution Heatmap generated and saved!")

if __name__ == "__main__":
    main()