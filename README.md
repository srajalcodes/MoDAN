# MoDAN: Modality-Disentangled Attention Network for Topology-Free Drug–Drug Interaction Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

MoDAN (Modality-Disentangled Attention Network) is a multimodal deep learning framework for drug–drug interaction (DDI) prediction under strict inductive and zero-shot evaluation settings.

Unlike graph-based methods that rely on interaction-network topology, MoDAN operates exclusively on intrinsic drug properties and therefore remains applicable to previously unseen compounds where network information is unavailable.

The framework integrates:
* **ChemBERTa** molecular embeddings
* **ESM-2** protein sequence embeddings
* **BioBERT** biological pathway embeddings

through a modality-disentangled gated cross-attention architecture that dynamically filters noisy modalities while preserving complementary biological information.

---

## Key Results

| Dataset  | Evaluation Setting | ROC-AUC         |
| -------- | ------------------ | --------------- |
| DrugBank | S1 (Old-New)       | 0.8548 ± 0.0004 |
| DrugBank | S2 (New-New)       | 0.7378 ± 0.0015 |
| BIOSNAP  | Zero-Shot Transfer | 0.7296 ± 0.0094 |
| ZhangDDI | Zero-Shot Transfer | 0.7232 ± 0.0040 |

MoDAN substantially outperforms classical machine learning baselines while maintaining robust generalization to previously unseen drugs and external datasets.

---

## Repository Structure

```text
MoDAN-DDI/
│
├── scripts/
├── configs/
├── metadata/
├── docs/
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── models/
├── data/
├── tests/
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Installation

### Conda Environment

```bash
conda env create -f environment.yml
conda activate ddi_final
```

### Pip Installation

```bash
pip install -r requirements.txt
```

---

## Dataset Acquisition

This study utilizes DrugBank v5.1.21.

Due to DrugBank licensing restrictions, the original dataset cannot be redistributed through this repository.

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

This will generate:

* Canonical DDI dataset
* Cold Train partition
* S1 partition
* S2 partition

Dataset statistics:

* Unique drugs: 3,904
* Positive interactions: 1,238,380
* Negative interactions: 1,238,380
* Total pairs: 2,476,760

---

## Training MoDAN

Train the production model:

```bash
python scripts/train_production_model.py
```

The best model checkpoint is selected according to S2 ROC-AUC.

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

Fair deep-learning baseline:

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

## External Benchmark Evaluation

Benchmark transfer experiments:

```bash
python scripts/train_biomodal_onthefly.py
```

Strict zero-shot evaluation:

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

## Reproducibility

A complete reproduction guide is available in:

```text
docs/reproducibility.md
```

The repository includes:

* Environment specification
* Hyperparameters
* Dataset statistics
* Evaluation protocols
* Statistical validation procedures

---

## Model Availability

The trained MoDAN checkpoint, supplementary results, and reproducibility materials are available through Zenodo.

A Hugging Face model repository is also provided for inference and future benchmarking studies.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
Citation will be added upon publication.
```

---

## License

This repository is released for academic and research purposes.

DrugBank-derived datasets are not redistributed and remain subject to the original DrugBank licensing terms.
