import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- MoDAN ARCHITECTURE ---
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

    def extract_penultimate_tensor(self, drug_a, drug_b):
        chem_a, esm_a, bio_a = self.encode(drug_a)
        chem_b, esm_b, bio_b = self.encode(drug_b)
        I_chem, I_esm, I_bio = self.attn_chem(chem_a, chem_b), self.attn_esm(esm_a, esm_b), self.attn_bio(bio_a, bio_b)
        D_chem, D_esm, D_bio = chem_a - chem_b, esm_a - esm_b, bio_a - bio_b
        h = torch.cat([I_chem, I_esm, I_bio, D_chem, D_esm, D_bio], dim=-1)
        for i in range(8): 
            h = self.classifier[i](h)
        return h

class OnTheFlyDDIDataset(Dataset):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chemberta", required=True)
    parser.add_argument("--esm2", required=True)
    parser.add_argument("--biobert", required=True)
    parser.add_argument("--test_csv", default=r"dataset\test_cold_S2.csv")
    parser.add_argument("--model_path", default=r"FInal_model\best_biomodal_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading dictionaries and model...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    # Load 3000 random samples from S2
    df = pd.read_csv(args.test_csv).sample(3000, random_state=42).reset_index(drop=True)
    dataset = OnTheFlyDDIDataset(df, chem, esm, bio, c_dim, e_dim, b_dim)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    raw_chem, raw_esm, raw_bio, final_tensor, labels = [], [], [], [], []

    print("Extracting features...")
    with torch.no_grad():
        for drug_a, drug_b, label in tqdm(loader):
            raw_chem.append(torch.cat([drug_a[:, :c_dim], drug_b[:, :c_dim]], dim=-1).cpu().numpy())
            raw_esm.append(torch.cat([drug_a[:, c_dim:c_dim+e_dim], drug_b[:, c_dim:c_dim+e_dim]], dim=-1).cpu().numpy())
            raw_bio.append(torch.cat([drug_a[:, c_dim+e_dim:], drug_b[:, c_dim+e_dim:]], dim=-1).cpu().numpy())
            
            features = model.extract_penultimate_tensor(drug_a.to(device), drug_b.to(device))
            final_tensor.append(features.cpu().numpy())
            labels.extend(label.numpy())

    data_dict = {
        "Raw Chemistry (ChemBERTa)": np.vstack(raw_chem),
        "Raw Proteins (ESM-2)": np.vstack(raw_esm),
        "Raw Pathways (BioBERT)": np.vstack(raw_bio),
        "MoDAN Fused Representation": np.vstack(final_tensor)
    }
    labels = np.array(labels)

    print("\nCalculating UMAP...")
    sns.set_theme(style="white", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {0: '#4DBBD5', 1: '#E64B35'}

    for i, (title, features) in enumerate(data_dict.items()):
        # Print silhouette scores to terminal so you have them for the text, 
        # but do NOT plot them on the figure!
        sil_score = silhouette_score(features, labels, metric='cosine')
        print(f"  -> {title} | Silhouette: {sil_score:.3f}")
        
        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)
        
        # Plot Scatter 
        sns.scatterplot(
            x=embedding[:, 0], y=embedding[:, 1], 
            hue=labels, palette=colors, 
            alpha=0.6, s=15, edgecolor=None, ax=axes[i], legend=False
        )
        
        axes[i].set_title(title, fontweight='bold', pad=14)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        sns.despine(ax=axes[i], bottom=True, left=True)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4DBBD5', markersize=13, label='Non-Interacting'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E64B35', markersize=13, label='Interacting')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, prop={'weight': 'bold', 'size': 12})

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save in 1200 DPI for RSC Journals
    plt.savefig("Figure_S4_Exploratory_UMAP.pdf", dpi=1200, bbox_inches='tight')
    plt.savefig("Figure_S4_Exploratory_UMAP.png", dpi=1200, bbox_inches='tight')
    plt.savefig("Figure_S4_Exploratory_UMAP.tif", dpi=1200, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    print("\n✅ Saved 1200 DPI UMAP Grid: Figure_S4_Exploratory_UMAP.pdf / .png / .tif")

if __name__ == "__main__":
    main()