# Explainable Deep Learning for Colorectal Cancer Histopathological Classification
## Overview
This repository contains the code and experimental results for **“Explainable Deep Learning for Colorectal Cancer Histopathological Classification: A Comparative Evaluation and External Validation of CNN Models.”**

Six pretrained CNN architectures were evaluated: **ResNet-34, ResNet-50, DenseNet-121, Inception-V3, VGG-16, and EfficientNet-B0**. The study covers (1) 10 independent training runs, (2) test-time augmentation (TTA), (3) external validation, (4) statistical comparisons, and (5) Grad-CAM++ explainability analysis.

## Data Availability
The image datasets are publicly accessible and **are not included in this repository** because of their size.

| Dataset | Use | Download | Original source |
| --- | --- | --- | --- |
| EBHI-SEG | Training and internal evaluation (six classes) | [Kaggle](https://www.kaggle.com/datasets/orvile/ebhi-seg-colorectal-cancer) | [Figshare](https://doi.org/10.6084/m9.figshare.21540159.v1) |
| LC25000 | External validation (1,000 colon images: 500 benign and 500 adenocarcinoma) | [Kaggle](https://www.kaggle.com/datasets/javaidahmadwani/lc25000) | [Original repository](https://github.com/tampapath/lung_colon_image_set) |

### Required dataset folders
Download and extract the datasets locally, arranging the files as follows (folder names can be case-sensitive):

```text
data/
├── EBHI-SEG/
│   ├── Adenocarcinoma/
│   │   ├── image/
│   │   └── label/
│   ├── High-grade IN/
│   │   ├── image/
│   │   └── label/
│   ├── Low-grade IN/
│   │   ├── image/
│   │   └── label/
│   ├── Normal/
│   │   ├── image/
│   │   └── label/
│   ├── Polyp/
│   │   ├── image/
│   │   └── label/
│   └── Serrated adenoma/
│       ├── image/
│       └── label/
└── lung_colon_image_set/
    └── Test Set/
        ├── colon_aca/
        └── colon_n/
```

EBHI-SEG `image/` folders contain histopathology images; `label/` folders contain segmentation masks used for explainability assessment. LC25000 `colon_aca/` and `colon_n/` contain images for external validation. Confirm the exact EBHI-SEG class-folder spelling against the downloaded copy and configure dataset paths in the notebooks to match the local location.

**Data availability statement:** The EBHI-SEG and LC25000 datasets used in this study were obtained from the Kaggle links attached above (their original sources are also linked). No new histopathological image dataset was collected. The images are not redistributed here; researchers can download them from the linked sources subject to their access and reuse terms. Code and experimental results shared with this repository are described below.

## Repository Structure
```text
project/
├── README.md
├── notebooks/
│   ├── efficientnet_b0.ipynb
│   ├── resnet_34.ipynb
│   ├── resnet_50.ipynb
│   ├── densenet_121.ipynb
│   ├── inception_v3.ipynb
│   ├── vgg_16.ipynb
│   ├── statistical_evaluation.ipynb
│   ├── xai_evaluation_gradcam++.ipynb
│   ├── utils/
│   │   ├── cache_utils.py
│   │   ├── data_utils.py
│   │   ├── experiment_runner.py
│   │   └── ...
│   └── outputs/
│       ├── resnet34/
│       │   └── run_cache/
│       │       ├── run_0_seed_42.json
│       │       └── ...
│       ├── resnet50/
│       ├── densenet121/
│       ├── inceptionv3/
│       ├── vgg16/
│       └── efficientnetb0/
└── data/
```

## Reproducing the Experiments
1. Download both datasets and arrange them under `data/` as shown above.
2. Set the local dataset paths in the notebooks and install Python dependencies.
3. Run the six architecture notebooks to train and evaluate models across the configured random seeds.
4. Run the statistical evaluation and Grad-CAM++ explainability notebooks to reproduce the corresponding analysis.

**Cached runs:** Each architecture's `run_cache/` contains run-specific `.json` files storing the experiment results needed for subsequent analysis, according to the fields saved by the notebook. When a matching JSON cache exists, the loader skips retraining and returns the cached results. To retrain a run, remove or bypass its JSON cache.

## Key Findings
ResNet-34 was the best-performing model of the 6 CNNs compared.
| Evaluation | ResNet-34 mean accuracy (%) |
| --- | ---: |
| EBHI-SEG | 91.01 ± 0.94 |
| LC25000 | 85.25 ± 3.63 |

TTA significantly improved internal accuracy for five of six architectures and external accuracy for all six. Grad-CAM++ was evaluated for consistency, plausibility, and fidelity.

## Dataset Citations
- Shi et al. (2023). *EBHI-Seg: A novel enteroscope biopsy histopathological hematoxylin and eosin image dataset for image segmentation tasks.* https://doi.org/10.3389/fmed.2023.1114673
- Borkowski et al. (2019). *Lung and Colon Cancer Histopathological Image Dataset (LC25000).* https://arxiv.org/abs/1912.12142
