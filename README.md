# Flood Segmentation from Sentinel-1 SAR

Binary flood segmentation using a U-Net trained on the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset. The model takes Sentinel-1 SAR imagery (VV + VH polarizations) and outputs per-pixel flood masks.

**Input:** 2-band SAR (512x512, 10m resolution) | **Output:** Binary flood mask | **Architecture:** U-Net

## Dataset

Sen1Floods11 contains 446 hand-labeled chips across 11 countries, split into:
- **Train:** ~252 chips | **Val:** ~89 chips | **Test:** ~90 chips (10 countries)
- **Bolivia:** 15 chips held out entirely for generalization testing

Labels: `1` = flood, `0` = non-flood, `-1` = invalid (masked during training).

## EDA

Detailed exploratory data analysis with plots is in [**EDA/README.md**](EDA/README.md). Key takeaways:
- Severe class imbalance — flood is only ~9% of pixels
- SAR backscatter separates flood from land by >7 dB, consistent across countries
- Otsu auto-labels are unreliable; training uses hand-labeled chips only

## Model

Standard U-Net with 4 encoder levels (2 → 64 → 128 → 256 → 512 → 1024). Combined Dice + BCE loss with invalid pixel masking. Trained with Adam (lr=1e-3), ReduceLROnPlateau scheduler, and early stopping (patience=10).

## Notebooks

| Notebook | Description |
|----------|-------------|
| `EDA/eda.ipynb` | Exploratory data analysis |
| `fixed_db_threshold.ipynb` | Best classical baseline — fixed VV threshold at -13.45 dB |
| `experiments/image_processing_baselines.ipynb` | Classical CV baselines (Otsu, adaptive, K-Means, morphological) |
| `experiments/remote_sensing_baselines.ipynb` | SAR-specific RS baselines (cross-pol ratio, NDFI, K-I, Lee filter, GLCM) |
| `model_training.ipynb` | Local training script |
| `model_training_colab.ipynb` | Colab version with GDrive checkpointing |

## Running on Colab

The Colab notebook downloads the dataset from GCS and saves checkpoints to Google Drive. If the runtime disconnects, re-run all cells — training resumes automatically from the last checkpoint.

## Project Structure

```
.
├── EDA/
│   ├── eda.ipynb       # EDA notebook
│   ├── README.md       # Detailed EDA writeup with plots
│   └── assets/         # EDA figures
├── experiments/
│   ├── image_processing_baselines.ipynb  # Classical CV baselines
│   └── remote_sensing_baselines.ipynb    # SAR remote sensing baselines
├── fixed_db_threshold.ipynb              # Best classical baseline
├── model_training.ipynb
├── model_training_colab.ipynb
└── Sen1Floods11/       # Dataset (git-ignored)
```
