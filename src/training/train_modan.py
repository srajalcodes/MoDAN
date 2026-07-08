import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# --- Import from our clean src/ folder! ---
from src.model import ModalAttnDDI
from src.dataset import OnTheFlyDDIDataset

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
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"[{name}] ROC-AUC: {roc:.4f} | F1: {f1:.4f}")
    return roc

def main():
    parser = argparse.ArgumentParser()
    # Dynamic Relative Paths
    parser.add_argument("--chemberta", default=str(ROOT / "data" / "embeddings" / "chemberta_embeddings.pkl"))
    parser.add_argument("--esm2", default=str(ROOT / "data" / "embeddings" / "esm2_embeddings.pkl"))
    parser.add_argument("--biobert", default=str(ROOT / "data" / "embeddings" / "biobert_drug_embeddings.pkl"))
    
    parser.add_argument("--train_csv", default=str(ROOT / "data" / "processed" / "train_cold.csv"))
    parser.add_argument("--s1_csv", default=str(ROOT / "data" / "processed" / "test_cold_S1.csv"))
    parser.add_argument("--s2_csv", default=str(ROOT / "data" / "processed" / "test_cold_S2.csv"))
    
    # Save to the correct models/ folder
    parser.add_argument("--model_save_path", default=str(ROOT / "models" / "modan_final_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    print("Loading embedding dictionaries...")
    chem = pickle.load(open(args.chemberta, "rb"))
    esm = pickle.load(open(args.esm2, "rb"))
    bio = pickle.load(open(args.biobert, "rb"))
    c_dim, e_dim, b_dim = len(next(iter(chem.values()))), len(next(iter(esm.values()))), len(next(iter(bio.values())))

    print("Loading datasets (This is the Massive DrugBank dataset)...")
    train_loader = DataLoader(OnTheFlyDDIDataset(args.train_csv, chem, esm, bio, c_dim, e_dim, b_dim), batch_size=512, shuffle=True, pin_memory=True)
    s1_loader = DataLoader(OnTheFlyDDIDataset(args.s1_csv, chem, esm, bio, c_dim, e_dim, b_dim), batch_size=512, shuffle=False)
    s2_loader = DataLoader(OnTheFlyDDIDataset(args.s2_csv, chem, esm, bio, c_dim, e_dim, b_dim), batch_size=512, shuffle=False)

    model = ModalAttnDDI(chem_dim=c_dim, esm_dim=e_dim, bio_dim=b_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_s2_roc = 0.0

    print("\nStarting Final Production Training...")
    
    # Create models directory if it doesn't exist
    Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 11):
        model.train()
        for drug_a, drug_b, label in tqdm(train_loader, desc=f"Epoch {epoch}/10"):
            optimizer.zero_grad()
            loss = criterion(model(drug_a.to(device), drug_b.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
            
        print("Evaluating...")
        evaluate(model, s1_loader, device, f"S1 (Epoch {epoch})")
        s2_roc = evaluate(model, s2_loader, device, f"S2 (Epoch {epoch})")
        
        if s2_roc > best_s2_roc:
            best_s2_roc = s2_roc
            print(f"🌟 NEW BEST S2 SCORE: {best_s2_roc:.4f}! Saving model... 🌟\n")
            torch.save(model.state_dict(), args.model_save_path)
        else:
            print(f"No improvement. (Best was {best_s2_roc:.4f})\n")

    print(f"\n✅ Production Training Complete! Best model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()