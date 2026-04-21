# Flood Segmentation from Sentinel-1 SAR

Binary flood segmentation on the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset using deep learning. Models take Sentinel-1 SAR imagery (VV + VH polarizations) and output per-pixel flood masks.

**Input:** 2-band SAR (512×512, 10m resolution) | **Output:** Binary flood mask | **Models:** U-Net (ResNet-34) · SegFormer (MiT-B2)

## Dataset

Sen1Floods11 contains 446 hand-labeled chips across 11 countries, split into:
- **Train:** 252 chips | **Val:** 89 chips | **Test:** 90 chips (10 countries)
- **Bolivia:** 15 chips held out entirely for out-of-distribution generalization testing

Labels: `1` = flood, `0` = non-flood, `-1` = invalid (masked during training and evaluation).

## EDA

Detailed exploratory data analysis with plots is in [**EDA/README.md**](EDA/README.md). Key takeaways:
- Severe class imbalance — flood is only ~9% of valid pixels
- SAR backscatter separates flood from land by >7 dB, consistent across countries
- Otsu auto-labels are unreliable; training uses hand-labeled chips only

## Models

### U-Net (ResNet-34 backbone, SMP)
UNet decoder with a pretrained ResNet-34 encoder from `segmentation-models-pytorch`. Combined Dice + BCE loss with invalid pixel masking. Trained with Adam (lr=3e-3), AMP, gradient clipping, ReduceLROnPlateau scheduler, and early stopping.

### SegFormer (MiT-B2, HuggingFace)
True SegFormer architecture — hierarchical Mix Transformer encoder with a lightweight all-MLP decoder. Logits upsampled 4× from H/4 back to full resolution. Trained with AdamW (lr=6e-5), AMP, gradient clipping, and early stopping.

## Results

### Test Set (90 chips, 10 countries)

| Model | IoU | F1 | Precision | Recall | Accuracy | Best Epoch | Params |
|---|---|---|---|---|---|---|---|
| U-Net (ResNet-34) | 0.6456 | 0.7847 | 0.8166 | 0.7551 | 0.9482 | 35 | 24.4M |
| SegFormer (MiT-B2) | **0.6680** | **0.8009** | 0.8111 | **0.7910** | **0.9508** | 29 | 27.3M |

### Bolivia Held-Out (15 chips, unseen country)

| Model | IoU | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| U-Net (ResNet-34) | 0.4808 | 0.6494 | **0.8414** | 0.5287 | 0.9094 |
| SegFormer (MiT-B2) | **0.6828** | **0.8115** | 0.7655 | **0.8635** | **0.9364** |

### Generalization Gap (Bolivia − Test IoU)

| Model | Test IoU | Bolivia IoU | Gap |
|---|---|---|---|
| U-Net (ResNet-34) | 0.6456 | 0.4808 | **−0.165** |
| SegFormer (MiT-B2) | 0.6680 | 0.6828 | **+0.015** |

SegFormer generalizes significantly better to the unseen Bolivia distribution — its Bolivia IoU actually *exceeds* its test IoU, while UNet degrades by 16.5 points.

### Per-Chip Extremes (Test Set)

| | U-Net | SegFormer |
|---|---|---|
| **Best chip** | Sri-Lanka_534068 (IoU=0.993) | Paraguay_34417 (IoU=1.000) |
| **Worst chips** | Ghana_97059, Ghana_53713, Ghana_83483 (IoU=0.000) | Ghana_97059, Ghana_53713, Ghana_83483 (IoU=0.000) |

Both models struggle on the same Ghana chips, suggesting those chips are inherently difficult (e.g., dry-season SAR response similar to standing water).

### Training Setup

| | U-Net | SegFormer |
|---|---|---|
| GPU | NVIDIA A100-SXM4-40GB | NVIDIA L4 |
| Batch size | 32 | 16 |
| Optimizer | Adam | AdamW |
| LR | 3e-3 | 6e-5 |
| Stopped at epoch | 50 | 44 |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `EDA/eda.ipynb` | Exploratory data analysis |
| `fixed_db_threshold.ipynb` | Best classical baseline — fixed VV threshold at -13.45 dB |
| `experiments/image_processing_baselines.ipynb` | Classical CV baselines (Otsu, adaptive, K-Means, morphological) |
| `experiments/remote_sensing_baselines.ipynb` | SAR-specific RS baselines (cross-pol ratio, NDFI, K-I, Lee filter, GLCM) |
| `model_training.ipynb` | Local training script |
| `model_training_colab.ipynb` | U-Net Colab training with GDrive checkpointing |
| `model_training_colab_result.ipynb` | U-Net training run with full outputs |
| `segformer_training_colab.ipynb` | SegFormer Colab training with GDrive checkpointing |
| `segformer_training_colab_result.ipynb` | SegFormer training run with full outputs |

## Running on Colab

Both Colab notebooks download the dataset from GCS and save checkpoints to Google Drive. If the runtime disconnects, re-run all cells — training resumes automatically from the last checkpoint.

## Streamlit Demo App

An interactive web app lets you select any of the 12 curated test chips (or upload your own), and compare all three models side by side.

### Setup

```bash
cd app
pip install -r requirements.txt
```

Set your HuggingFace repo ID in the sidebar (the repo must contain `unet_flood_best.pt` and `segformer_flood_best.pt`).

### Run

```bash
streamlit run app.py
```

### Features
- **Sample chips** — 12 curated chips (best, worst, and average from 7 countries)
- **Upload your own** — any `_S1Hand.tif` from the Sen1Floods11 dataset
- **Three models** — Image Processing (VV threshold), U-Net (ResNet-34), SegFormer (MiT-B2)
- **Per-model error maps** — correct / false positive / false negative visualisation
- **Metrics table** — IoU, F1, Precision, Recall, Accuracy with best value highlighted
- **IoU bar chart** — quick visual comparison

### Storage

| Asset | Location |
|---|---|
| 12 curated test chips | `app/sample_data/` (in repo, ~15 MB) |
| U-Net weights | HuggingFace Hub (`unet_flood_best.pt`) |
| SegFormer weights | HuggingFace Hub (`segformer_flood_best.pt`) |
| Image processing baseline | Code only — no weights |

## Project Structure

```
.
├── app/
│   ├── app.py                                 # Streamlit app
│   ├── inference.py                           # Model loading + prediction
│   ├── requirements.txt
│   └── sample_data/
│       ├── S1/                                # 12 curated SAR chips
│       └── Labels/                            # Matching ground-truth labels
├── EDA/
│   ├── eda.ipynb                              # EDA notebook
│   ├── README.md                              # Detailed EDA writeup with plots
│   └── assets/                                # EDA figures
├── experiments/
│   ├── image_processing_baselines.ipynb       # Classical CV baselines
│   └── remote_sensing_baselines.ipynb         # SAR remote sensing baselines
├── fixed_db_threshold.ipynb                   # Best classical baseline
├── model_training.ipynb                       # Local training script
├── model_training_colab.ipynb                 # U-Net Colab notebook
├── model_training_colab_result.ipynb          # U-Net results
├── segformer_training_colab.ipynb             # SegFormer Colab notebook
├── segformer_training_colab_result.ipynb      # SegFormer results
└── Sen1Floods11/                              # Dataset (git-ignored)
```
