import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Suppress RDKit warnings for messy SMILES
warnings.filterwarnings("ignore")


# Modern Academic Styling
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update({'font.family': 'sans-serif', 'figure.dpi': 300})

def get_fingerprint(smiles):
    """ Converts a SMILES string into a 2048-bit Morgan Fingerprint """
    if pd.isna(smiles): 
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except:
        return None

def main():
    print("Loading datasets...")

    nodes_path = METADATA_DIR / "final_drug_nodes.csv"
    train_path = PROCESSED_DATA_DIR / "train_cold.csv"
    s2_path = PROCESSED_DATA_DIR / "test_cold_S2.csv"


    for file in [nodes_path, train_path, s2_path]:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found:\n{file}")


    nodes_df = pd.read_csv(nodes_path)
    smiles_dict = dict(zip(nodes_df["drugbank_id"], nodes_df["smiles"]))


    train_df = pd.read_csv(train_path)
    s2_df = pd.read_csv(s2_path)

    seen_drugs = set(train_df["drug_A"]).union(train_df["drug_B"])
    unseen_drugs = set(s2_df["drug_A"]).union(s2_df["drug_B"])

    print(f"Total Seen Drugs (Train): {len(seen_drugs)}")
    print(f"Total Unseen Drugs (S2): {len(unseen_drugs)}")

    # 1. Calculate Fingerprints for the Training Set
    print("\nCalculating Morgan Fingerprints for Training Set...")
    seen_fps = []
    for d in tqdm(seen_drugs, desc="Train Drugs"):
        fp = get_fingerprint(smiles_dict.get(d, np.nan))
        if fp is not None:
            seen_fps.append(fp)

    # 2. Find the Maximum Similarity for every Unseen Drug
    print("\nCalculating Maximum Tanimoto Similarity for Unseen S2 Drugs...")
    max_similarities = []
    
    for d in tqdm(unseen_drugs, desc="Unseen S2 Drugs"):
        fp = get_fingerprint(smiles_dict.get(d, np.nan))
        if fp is not None and len(seen_fps) > 0:
            # Compare this unseen drug to EVERY drug in the training set
            sims = DataStructs.BulkTanimotoSimilarity(fp, seen_fps)
            max_sim = max(sims) # Get the closest neighbor
            max_similarities.append(max_sim)

    # 3. Calculate Checklist Metrics
    mean_sim = np.mean(max_similarities)
    median_sim = np.median(max_similarities)
    max_overall = np.max(max_similarities)

    print("\n" + "="*50)
    print("🧪 SIMILARITY LEAKAGE ANALYSIS RESULTS 🧪")
    print("="*50)
    print(f"Total Small Molecules Evaluated: {len(max_similarities)}")
    print(f"Mean Tanimoto Similarity:   {mean_sim:.4f}")
    print(f"Median Tanimoto Similarity: {median_sim:.4f}")
    print(f"Absolute Max Similarity:    {max_overall:.4f}")
    print("="*50)

# 4. Plot the Distribution
    plt.figure(figsize=(7, 5))
    
    # Changed color to a deeper, more academic steel blue
    sns.histplot(max_similarities, bins=40, kde=True, color='#2C7BB6', edgecolor='black', alpha=0.7)
    
    # Add vertical lines for Mean and Median
    plt.axvline(mean_sim, color='#E64B35', linestyle='--', linewidth=2.5, label=f'Mean: {mean_sim:.2f}')
    plt.axvline(median_sim, color='Black', linestyle=':', linewidth=2.5, label=f'Median: {median_sim:.2f}')
    
    # Removed shaded region, removed text box, removed title!

    plt.xlabel('Maximum Tanimoto Similarity to Training Set', fontweight='bold')
    plt.ylabel('Number of Unseen Drugs', fontweight='bold')
    plt.xlim(0, 1.0)
    
    plt.legend(frameon=True, fontsize=12)
    sns.despine()

    plt.tight_layout()
    
    # Save in ultra-high resolution (1200 DPI) for RSC Print Standards
    plt.savefig('Figure_S3_Tanimoto_Similarity.pdf', dpi=1200, bbox_inches='tight')
    plt.savefig('Figure_S3_Tanimoto_Similarity.png', dpi=1200, bbox_inches='tight')
    plt.savefig('Figure_S3_Tanimoto_Similarity.tif', dpi=1200, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    
    print("\n✅ Generated 'Figure_S3_Tanimoto_Similarity' in 1200 DPI!")
    print("Clean, academic format with no title/text boxes.")

if __name__ == "__main__":
    main()