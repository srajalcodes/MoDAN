import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Ultra-Modern Scientific Theme ---
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.dpi': 300
})

def get_top_entities(features_dict, vocab_map, category, top_k=10):
    """Finds the most frequent biological entities across all drugs."""
    reverse_map = {idx: name for name, idx in vocab_map.items()}
    total_counts = np.zeros(len(vocab_map))
    
    for drug_id, feats in features_dict.items():
        total_counts += feats[category]
        
    top_indices = np.argsort(total_counts)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        # Some JSON dict keys might be strings, make sure we use integer indices
        results.append((reverse_map[idx], int(total_counts[idx])))
    return results

def main():
    print("Loading biological data...")
    
    # 1. EXACT PATHS YOU PROVIDED
    features_path = r"C:\Users\st735\OneDrive - Shiv Nadar Institution of Eminence\Documents\CODE\DDI\Final_model ddi\outputs\biological_features_universal.json"
    vocab_path = r"C:\Users\st735\OneDrive - Shiv Nadar Institution of Eminence\Documents\CODE\DDI\Final_model ddi\outputs\biological_vocab.json"
    drug_meta_path = r"dataset\final_drug_nodes.csv"
    
    # 2. Load the Vocab Dictionary
    with open(vocab_path, "r") as f:
        vocabs = json.load(f)
        
    # 3. Load the Features (and convert JSON lists back to numpy arrays)
    with open(features_path, "r") as f:
        features_raw = json.load(f)
        
    features = {}
    for drug_id, feats in features_raw.items():
        features[drug_id] = {
            'targets': np.array(feats['targets']),
            'enzymes': np.array(feats['enzymes']),
            'transporters': np.array(feats['transporters'])
        }

    # Extract the Top 10 most common metabolizing enzymes
    print("Calculating enzyme frequencies...")
    top_enzymes = get_top_entities(features, vocabs['enzymes'], 'enzymes', top_k=10)
    
    names = [x[0] for x in top_enzymes]
    counts = [x[1] for x in top_enzymes]

    print("\nTop 5 Drug-Metabolizing Enzymes found:")
    for name, count in top_enzymes[:5]: 
        print(f"  {count} drugs -> {name}")

    # =============================================================================
    # DRAW THE CHART
    # =============================================================================
    print("\nGenerating high-resolution publication chart...")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Use a professional dark blue/teal academic palette
    sns.barplot(x=counts, y=names, palette="mako", edgecolor=".2", ax=ax)
    
    # Add the exact numbers to the end of each bar
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + max(counts)*0.01, p.get_y() + p.get_height()/2. + 0.1, 
                f'{int(width)}', ha="left", va="center", fontweight='bold', fontsize=11)

    # Styling
    ax.set_title('Top 10 Targeted Pharmacokinetic Enzymes in DrugBank', pad=15)
    ax.set_xlabel('Number of Interacting Drugs')
    ax.set_ylabel('Biological Pathway / Enzyme')
    ax.set_xlim(0, max(counts) * 1.15) # Leave space for the text labels
    
    sns.despine() # Clean borders
    plt.tight_layout()
    
    # Create Figures folder if it doesn't exist
    os.makedirs("Figures", exist_ok=True)
    
    # Save as 1200 DPI PDF and PNG
    plt.savefig(r"Figures\biological_distribution.pdf", dpi=1200, bbox_inches='tight')
    plt.savefig(r"Figures\biological_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(r"Figures\biological_distribution.tif", dpi=1200, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    
    print("✅ Successfully saved to Figures/biological_distribution.pdf!")

if __name__ == "__main__":
    main()