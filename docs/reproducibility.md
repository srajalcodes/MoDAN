# Reproducibility Guide

## Overview

This repository accompanies the manuscript:

**"MoDAN: An Interpretable Modality-Disentangled Attention Network for Zero-Shot Drug–Drug Interaction Prediction"**

The repository provides the complete source code required to reconstruct the dataset, train the proposed model, evaluate its performance, reproduce the reported figures and tables, and validate the experimental results presented in the manuscript.

To facilitate reproducibility while complying with DrugBank licensing restrictions, the reproducibility resources are distributed across **GitHub** and **Zenodo**.

---

# Repository Resources

## GitHub Repository

The complete source code is available at:

**https://github.com/srajalcodes/MoDAN**

The GitHub repository contains:

- Dataset reconstruction pipeline
- Training scripts
- Baseline implementations
- Evaluation scripts
- Statistical analysis scripts
- Visualization utilities
- Configuration files
- Documentation

---

## Zenodo Archive

The complete reproducibility package is available at:

**https://doi.org/10.5281/zenodo.21221081**

The Zenodo archive contains:

- Pre-trained MoDAN model weights (`best_biomodal_model.pt`)
- ChemBERTa embeddings
- ESM-2 embeddings
- BioBERT embeddings
- DrugBank evaluation splits
- BIOSNAP benchmark splits
- ZhangDDI benchmark splits

The original DrugBank XML database is **not included** because it is distributed under the DrugBank Academic License.

---

# Environment Setup

## Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate ddi_final
```

## Option 2: Pip

```bash
pip install -r requirements.txt
```

---

# Dataset Acquisition

This study utilizes **DrugBank Version 5.1 (schema version 5.1.13; exported on 2025-01-02).**

The original DrugBank XML database cannot be redistributed through this repository due to DrugBank licensing restrictions.

Researchers wishing to reproduce the complete dataset should obtain DrugBank directly through the official DrugBank Academic Licensing Program.

Additional biological annotations are obtained from:

- UniProt
- PubChem

---

# Dataset Reconstruction

After obtaining DrugBank, reconstruct the canonical dataset using:

```bash
python src/preprocessing/build_canonical_dataset.py
```

This pipeline generates:

- Canonical DrugBank DDI dataset
- Training partition
- S1 (Old–New) evaluation split
- S2 (New–New) evaluation split

Dataset statistics:

| Statistic | Value |
|-----------|------:|
| Unique drugs | 3,904 |
| Positive interactions | 1,238,380 |
| Negative interactions | 1,238,380 |
| Total interaction pairs | 2,476,760 |

---

# Training MoDAN

Train the proposed MoDAN model using:

```bash
python src/training/train_modan.py
```

The training pipeline automatically selects the best-performing checkpoint according to the validation ROC-AUC on the S2 inductive evaluation protocol.

Expected output:

```text
best_biomodal_model.pt
```

---

# Baseline Models

## Dual-Modal Baselines

```bash
python src/training/train_logistic_regression.py

python src/training/train_xgboost.py
```

## Tri-Modal Baseline

```bash
python src/training/train_xgboost_multimodal.py
```

## Deep Learning Baseline

```bash
python src/training/train_mlp_baseline.py
```

---

# Ablation Studies

## Modality Ablation

```bash
python src/training/run_ablation_study.py
```

## Architectural Ablation

```bash
python src/analysis/analyze_gate_ablation.py
```

---

# External Benchmark Evaluation

## Cross-Dataset Transfer Learning

```bash
python src/training/train_modan_bimodal.py
```

## Strict Zero-Shot Transfer

```bash
python src/evaluation/evaluate_zero_shot_transfer.py
```

---

# Statistical Analysis

## Bootstrap Confidence Intervals

```bash
python src/evaluation/compute_evaluation_metrics.py
```

## Statistical Significance Testing

```bash
python src/evaluation/statistical_significance_tests.py
```

## Calibration Analysis

```bash
python src/visualization/generate_calibration_plots.py
```

---

# Figure Generation

## UMAP Visualization

```bash
python src/visualization/generate_umap_visualization.py
```

## Biological Annotation Analysis

```bash
python src/analysis/analyze_biological_annotations.py
```

## Model Interpretability Analysis

```bash
python src/analysis/analyze_model_interpretability.py
```

---

# Expected Outputs

Following the procedures described above should reproduce the experimental results reported in the accompanying manuscript.

Minor numerical differences may occur because of hardware characteristics, software versions, floating-point arithmetic, and random initialization.

For maximum reproducibility, we recommend:

- Using the supplied `environment.yml`
- Using the published Zenodo reproducibility package
- Using the same DrugBank release described above
- Preserving the provided train/test partitions

---

# Reproducibility Statement

All experiments reported in the manuscript can be reproduced using:

- The GitHub source code repository
- The Zenodo reproducibility archive
- A licensed copy of DrugBank Version 5.1 (schema version 5.1.13)

The repository has been organized to separate source code (GitHub) from large research artifacts (Zenodo), following common practices for computational biology and machine learning research while respecting DrugBank licensing requirements.
