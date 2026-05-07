# AGCFNet: Attention-Gated Convolutional Fusion Network for Alzheimer's Disease Staging

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ADNI](https://img.shields.io/badge/Data-ADNI-green.svg)](https://adni.loni.usc.edu/)

> **Paper:** *AGCFNet: Attention-Gated Convolutional Fusion Network for Alzheimer's Disease Staging from Structural MRI and Clinical Biomarkers*  

---

## Overview

AGCFNet is a multimodal deep learning framework for **three-way Alzheimer's Disease staging** (Cognitively Normal / Mild Cognitive Impairment / AD) from baseline T1-weighted structural MRI combined with routine clinical biomarkers (age, sex, MMSE, APOE4).

### Key Results on ADNI (n = 970)

| Metric | Value |
|---|---|
| Macro AUC | **0.890 ± 0.000** |
| Tuned Balanced Accuracy | **79.7%** |
| Cohen's κ | **0.595** |
| MCI-to-AD Conversion AUC (longitudinal subset) | **0.80** |

### Architecture at a Glance

```
T1-MRI Volume (96×112×96)
      │
      ▼
3-D MedNeXt Encoder (4 stages, GRN + stochastic depth)
      │                    │
      │             Stage-3 activations
      │                    │
      │         Persistence Graph Builder
      │                    │
      │              2-layer GCN
      │                    │
  192-d fimg ──── concat ──── 128-d fgrp  →  320-d fvis
                                                   │
                              Clinical Features ───┤ Gated Cross-Modal Attention
                              (Age, Sex, MMSE,     │
                               APOE4)              │
                                             3-class logits (CN / MCI / AD)
```

**Three cooperating modules:**
1. **3-D MedNeXt Encoder** — depthwise 3-D convolutions with Global Response Normalisation (GRN) and stochastic depth for numerically stable AMP/BF16 training on whole-brain volumes.
2. **Persistence-Inspired Graph Builder** — zero-parameter topological graph from Stage-3 activation maps via 6-connected union-find filtration, processed by a lightweight 2-layer GCN.
3. **Gated Cross-Modal Attention Fusion** — scalar gate dynamically balances high-capacity imaging features against low-dimensional clinical features at inference time.

---

## Repository Structure

```
AGCFNet/
├── adni_alzheimer.ipynb      # Main notebook: preprocessing → training → evaluation
├── AGCFNet_ADNI.pdf          # Companion paper describing AGCFNet
├── requirements.txt          # Python dependencies
├── README.md
└── LICENSE
```

---

## Dataset

Data must be obtained independently from the [ADNI database](https://adni.loni.usc.edu/) (registration required).

**Cohort used in the paper:**

| Split | CN | MCI | AD | Total |
|---|---|---|---|---|
| Train (80%) | 229 | 379 | 168 | 776 |
| Validation (20%) | 59 | 99 | 36 | 194 |
| **Total** | **288** | **474** | **208** | **970** |

**Expected directory layout after download:**

```
ADNI/
├── ADNI_part1_fixed/ADNI/      # DICOM files (part 1)
├── ADNI_part5_fixed/ADNI/      # DICOM files (part 5)
└── Data___Database/
    └── ADNIMERGE2/ADNIMERGE2/data/
        ├── DXSUM.rda
        ├── PTDEMOG.rda
        ├── MMSE.rda
        └── APOERES.rda
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AGCFNet.git
cd AGCFNet
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependency

`dcm2niix` is required for DICOM → NIfTI conversion:

```bash
# Ubuntu / Debian
sudo apt-get install dcm2niix

# macOS (Homebrew)
brew install dcm2niix
```

---

## Running the Notebook

Open `adni_alzheimer.ipynb` in Jupyter or on Kaggle/Colab (GPU recommended — the notebook was developed on Kaggle with a P100/T4).

**Notebook sections:**

| Section | Description |
|---|---|
| 1. Environment setup | Installs packages, sets seeds, defines `Config` |
| 2. Clinical data loading | Parses ADNIMERGE `.rda` files, builds feature table |
| 3. DICOM → NIfTI conversion | Batch conversion with disk-space guard |
| 4. MRI preprocessing | TorchIO pipeline: orientation → rescale → resample → crop → z-normalise |
| 5. Dataset & augmentation | `ADNIDataset`, mixup, 6 stochastic augmentations |
| 6. Model definition | `MedNeXtEncoder`, `PersistenceInspiredGraphBuilder`, `AGCFNet` |
| 7. Training | AdamW + warm-up cosine annealing, early stopping, AMP (BF16) |
| 8. Evaluation | ROC / PR curves, confusion matrix, Cohen's κ, calibration |
| 9. Interpretability | GradCAM saliency maps, permutation importance, counterfactual probing |
| 10. Longitudinal analysis | Frozen-model conversion prediction, trajectory tracking |

---

## Interpretability

### GradCAM Saliency Maps
Activation overlays consistently highlight the **hippocampus, entorhinal cortex, and inferior temporal gyrus** for AD subjects — well-established sites of early tau pathology and atrophy.

### Clinical Feature Importance (Permutation-based)
| Feature | ΔAUC |
|---|---|
| MMSE | +0.295 |
| APOE4 | +0.037 |
| Age | +0.036 |
| Sex | −0.000 |

### Counterfactual Analysis
- **MMSE**: P(AD) drops and P(CN) rises monotonically as MMSE increases; curves cross near MMSE ≈ 24 (standard clinical impairment threshold).
- **APOE4**: P(AD) increases monotonically with allele count.

---

## Comparison with Clinical Baselines

| Model | Input | Balanced Acc. | Eval |
|---|---|---|---|
| Random Forest | Clinical only | 0.740 ± 0.020 | 5-fold |
| SVM | Clinical only | 0.724 ± 0.018 | 5-fold |
| GBM | Clinical only | 0.744 ± 0.017 | 5-fold |
| **AGCFNet** | **MRI + Clinical** | **0.797 (tuned)** | Single run |

---


---

## Data Acknowledgement

Data used in preparation of this work were obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** database ([adni.loni.usc.edu](https://adni.loni.usc.edu)). The investigators within ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at:  
http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

> **Note:** The ADNI dataset itself is subject to separate data use agreements. You must apply for access at [adni.loni.usc.edu](https://adni.loni.usc.edu) before using the data.
