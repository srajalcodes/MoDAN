# Reproducibility Guide

## Overview

This repository accompanies the MoDAN framework for topology-free multimodal drug-drug interaction prediction.

The repository contains all code required to reconstruct the dataset, train the model, evaluate performance, reproduce figures, and regenerate the tables reported in the manuscript.

---

## Environment Setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate ddi_final
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

---

## Data Acquisition

The study uses DrugBank v5.1.21.

DrugBank data cannot be redistributed through this repository due to licensing restrictions.

Researchers should obtain DrugBank directly through the official academic licensing program.

Additional biological annotations are retrieved from:

* UniProt
* PubChem

---

## Dataset Reconstruction

After obtaining DrugBank:

```bash
python scripts/build_dataset.py
```

Expected outputs:

* Canonical DDI dataset
* Train partition
* S1 partition
* S2 partition

Dataset statistics:

* Unique drugs: 3904
* Positive interactions: 1,238,380
* Negative interactions: 1,238,380

---

## Training MoDAN

Train the production model:

```bash
python scripts/train_production_model.py
```

Outputs:

```text
best_biomodal_model.pt
```

The best model is selected according to S2 ROC-AUC.

---

## Baseline Models

Dual-modal baselines:

```bash
python scripts/train_lr_onthefly.py

python scripts/train_xgboost_onthefly.py
```

Tri-modal baselines:

```bash
python scripts/train_model1_onthefly.py
```

MLP baseline:

```bash
python scripts/train_mlp_baseline.py
```

---

## Ablation Studies

Modality ablation:

```bash
python scripts/train_ablation.py
```

Architectural ablation:

```bash
python scripts/ablate_the_gate.py
```

---

## Transfer Learning

Benchmark evaluation:

```bash
python scripts/train_biomodal_onthefly.py
```

Strict zero-shot transfer:

```bash
python scripts/zero_shot_transfer.py
```

---

## Statistical Analysis

Bootstrap confidence intervals:

```bash
python scripts/calculate_rigorous_metrics.py
```

Permutation testing:

```bash
python scripts/calculate_p_values.py
```

Calibration analysis:

```bash
python scripts/generate_calibration.py
```

---

## Figure Generation

UMAP visualization:

```bash
python scripts/generate_advanced_umaps.py
```

Biological quality analysis:

```bash
python scripts/check_biological_quality.py
```

Interpretability analysis:

```bash
python scripts/extract_interpretability.py
```

---

## Expected Outputs

The reproduced results should closely match the values reported in the manuscript, with minor numerical differences possible due to platform-specific implementation details and software versions.

For exact reproduction, use the supplied environment.yml file.
