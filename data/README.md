# Data Availability

The original DrugBank database is **not included** in this repository due to DrugBank licensing restrictions.

This study was conducted using **DrugBank Version 5.1 (schema version 5.1.13; exported on 2025-01-02)**.

Researchers wishing to reproduce the complete dataset should obtain an academic license directly from DrugBank and download the corresponding XML database from the official DrugBank website.

After obtaining DrugBank, the canonical datasets used in this study can be reconstructed using:

```bash
python src/preprocessing/build_canonical_dataset.py
```

The reconstruction pipeline generates:

- Canonical Drug–Drug Interaction (DDI) dataset
- Training partition
- S1 (Old–New) inductive evaluation split
- S2 (New–New) inductive evaluation split

For reproducibility, pretrained model weights, multimodal embeddings, and benchmark evaluation splits are available through the Zenodo archive:

**https://doi.org/10.5281/zenodo.21221081**
