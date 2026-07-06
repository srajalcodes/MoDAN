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
MoDAN/
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── analysis/
│   ├── visualization/
│   ├── embeddings/
│   ├── model/
│   └── utils/
│
├── configs/
├── data/
├── docs/
├── metadata/
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
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
## Quick Start

```bash
git clone https://github.com/srajalcodes/MoDAN.git

cd MoDAN

conda env create -f environment.yml

conda activate ddi_final
```

Download the reproducibility package from Zenodo:

https://doi.org/10.5281/zenodo.21221081

Extract the archive into the project directory before running the training or evaluation scripts.

---

## Dataset Acquisition

This study utilizes DrugBank v5.1.13.

Due to DrugBank licensing restrictions, the original dataset cannot be redistributed through this repository.

Researchers should obtain DrugBank directly through the official academic licensing program.

Additional biological annotations are retrieved from:

* UniProt
* PubChem

---

## Data and Model Availability

The complete reproducibility package is divided between GitHub and Zenodo.

### GitHub Repository

This repository contains:

- Source code
- Training and evaluation scripts
- Configuration files
- Documentation
- Statistical analysis scripts
- Figure generation utilities

### Zenodo Archive

The complete reproducibility package is available at:

**DOI:** https://doi.org/10.5281/zenodo.21221081

The Zenodo archive contains:

- Pre-trained MoDAN model weights (`best_biomodal_model.pt`)
- ChemBERTa embeddings
- ESM-2 embeddings
- BioBERT embeddings
- DrugBank evaluation splits
- BIOSNAP evaluation splits
- ZhangDDI evaluation splits

Due to DrugBank licensing restrictions, the original DrugBank XML database is **not redistributed**. Researchers must obtain an academic license directly from DrugBank before running the dataset reconstruction pipeline.

## Dataset Reconstruction

After obtaining DrugBank:

```bash
python src/preprocessing/build_canonical_dataset.py
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
python src/training/train_modan.py
```

The best model checkpoint is selected according to S2 ROC-AUC.

---

## Baseline Models

Dual-modal baselines:

```bash
python src/training/train_logistic_regression.py
python src/training/train_xgboost.py
```

Tri-modal baselines:

```bash
python src/training/train_xgboost_multimodal.py
```

Fair deep-learning baseline:

```bash
python src/training/train_mlp_baseline.py
```

---

## Ablation Studies

Modality ablation:

```bash
python src/training/run_ablation_study.py
```

Architectural ablation:

```bash
python src/analysis/analyze_gate_ablation.py
```

---

## External Benchmark Evaluation

Benchmark transfer experiments:

```bash
python src/training/train_modan_bimodal.py
```

Strict zero-shot evaluation:

```bash
python src/evaluation/evaluate_zero_shot_transfer.py
```

---

## Statistical Analysis

Bootstrap confidence intervals:

```bash
python src/evaluation/compute_evaluation_metrics.py
```

Permutation testing:

```bash
python src/evaluation/statistical_significance_tests.py
```

Calibration analysis:

```bash
python src/visualization/generate_calibration_plots.py
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

The complete reproducibility package is available through Zenodo:

https://doi.org/10.5281/zenodo.21221081

The archive contains:

- Pre-trained model weights
- Multimodal embeddings
- Evaluation datasets
- Benchmark splits

The source code is maintained on GitHub.

---

## Citation

If you use this repository, please cite:

1. The accompanying manuscript (once published).

2. The Zenodo reproducibility archive:

Sharma D., Tiwari S., Singh J., Singh T.

*Dataset and Pre-Trained Weights for MoDAN.*

Zenodo.

https://doi.org/10.5281/zenodo.21221081


---

## License

This repository is released for academic and research purposes.

DrugBank-derived datasets are not redistributed and remain subject to the original DrugBank licensing terms.
