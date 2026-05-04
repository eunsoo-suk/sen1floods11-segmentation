# Distribution-Aware Cascaded MLOps Pipeline for SAR Flood Segmentation

A containerized, ClearML-orchestrated flood-mapping system that routes
each Sentinel-1 SAR chip to a fast classical detector or a deep model
based on the chip's VV-distribution shape. Built on the
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) benchmark.

**Live demo:** [Streamlit app](https://hardikkamboj-sen1floods11-segmentation-appapp-jni2fg.streamlit.app/)
| **Final report:** [`report/final_report.md`](report/final_report.md)
| **Reproducibility:** [`REPRODUCING.md`](REPRODUCING.md)

```bash
# One-command end-to-end reproduction
git clone <repo-url> && cd group_project
export CLEARML_API_ACCESS_KEY=... CLEARML_API_SECRET_KEY=...
make docker-demo
```

---

## What this is

Most flood-mapping systems — academic baselines and operational
pipelines (Copernicus EMS, GFM, HydroSAR, Cloud-to-Street) alike —
deploy a *single monolithic deep model* and ignore the strong
classical SAR thresholds that predate them. We build a hybrid cascade:

1. **Classical fast-pass** — fixed-dB threshold on VV, applied to every chip.
2. **Distribution-aware routing** — chips whose VV histograms are
   *bimodal* (Fisher discriminant ≥ τ) and *aligned* with the physics
   threshold are *trusted* to the classical method. Everything else is
   routed to the deep model.
3. **Deep refinement** — SegFormer MiT-B2 on the routed chips only.
4. **ClearML Pipeline** orchestrates the DAG; every stage is independently
   observable and queueable.

To our knowledge, this physics-grounded routing — pairing a non-learned
classical detector with a deep model in a production MLOps pipeline —
has not been published for SAR flood segmentation. Full literature
review in [`report/literature_survey.md`](report/literature_survey.md).

## System contributions (5 axes)

| Axis | What we deliver |
|---|---|
| **Availability** | Classical fallback (0.375 / 0.618 IoU on test / Bolivia) keeps the system serving when SegFormer is offline |
| **Reliability** | Per-chip `bimodality`, `alignment`, `frac_deep`, latency — all browsable as ClearML scalars |
| **Efficiency** | At τ=6, **3 % of test chips and 7 % of Bolivia chips skip the deep model** with ΔIoU = −0.002 |
| **Scalability** | ClearML Pipeline DAG with independently-queueable stages; routing and deep inference scale separately |
| **Robustness** | Bolivia OOD IoU **preserved or improved** (0.7087 cascade vs 0.7077 deep-only) |

Numbers anchored to ClearML benchmark task `26f0f478b9f64e7a9f18d8f92aff96f2`.

---

## Headline results (full Sen1Floods11)

### Models (per-chip, baselines for the cascade)

| Model | Test IoU | Bolivia IoU | Bolivia − Test |
|---|---|---|---|
| Classical (fixed −13.45 dB on VV) | 0.375 | 0.618 | +0.243 |
| U-Net (ResNet-34) | 0.646 | 0.481 | −0.165 |
| **SegFormer (MiT-B2)** | **0.652** | **0.708** | **+0.056** |

### Cascade (deep stage = SegFormer; sweep over bimodality τ)

| τ | Test IoU | Test frac_deep | Bolivia IoU | Bolivia frac_deep |
|---|---|---|---|---|
| 2 | 0.510 | 71.1 % | 0.663 | 73.3 % |
| 4 | 0.598 | 87.8 % | 0.663 | 73.3 % |
| **6 (sweet spot)** | **0.650** | **96.7 %** | **0.709** | **93.3 %** |
| 8 | 0.652 | 98.9 % | 0.709 | 93.3 % |
| 10+ (= deep-only) | 0.652 | 100 % | 0.708 | 100 % |

The sweet spot is selected by maximizing compute saving subject to
ΔIoU ≤ 0.005 of deep-only.

### Why the efficiency gain is modest on this dataset

Sen1Floods11 is curated for difficulty — most chips contain
confounders (vegetation, urban shadow, partial cover) that
suppress bimodality. Only ~3 % of chips meet the strict trust
criterion. **Operational SAR scenes** with large homogeneous
regions (open ocean, dense forest, desert) will see substantially
higher cascade savings — see [`report/final_report.md`](report/final_report.md)
§5 for discussion.

---

## Reproducing every reported number

Every number in the report is the output of a single `make` target,
logged to a specific ClearML Task. Hardware: Colab Pro+ with NVIDIA L4;
also runs CPU-only via Docker.

```bash
make setup                     # install pinned deps
make download-data             # ~2 GB Sen1Floods11 hand-labeled split
make download-model            # SegFormer checkpoint (~110 MB) from HF Hub
                               # OR: make train-segformer  (re-train end-to-end)
make calibrate                 # Figure A: empirical ambiguity band
make benchmark                 # Figure B + Figure C + system_comparison.md
make pipeline                  # cascade end-to-end on a sample scene
```

Or one shot inside Docker:

```bash
make docker-build && make docker-demo
```

Generated artifacts (mirrored as ClearML Task artifacts):

```
mlops/
├── calibration.json                       # empirical ambiguity band
├── figures/
│   ├── ambiguity_calibration.png          # Figure A
│   ├── figure_b_tradeoff.png              # Figure B
│   ├── figure_c_distribution_test.png     # Figure C (test)
│   └── figure_c_distribution_bolivia.png  # Figure C (Bolivia)
└── results/
    ├── benchmark.csv                      # raw per-strategy numbers
    └── system_comparison.md               # auto-generated headline table
```

ClearML Task IDs anchoring published numbers — see [`REPRODUCING.md`](REPRODUCING.md):

| Asset | Task ID |
|---|---|
| SegFormer training | `114283c1b1504969b5cb4fd67ac04bbf` |
| Empirical calibration | `29fd798900f543468ef042a00fc55093` |
| Cascade benchmark | `26f0f478b9f64e7a9f18d8f92aff96f2` |

---

## Repository structure

```
.
├── README.md                        # this file
├── REPRODUCING.md                   # canonical reproduction recipe
├── Dockerfile                       # CPU-default container; GPU swap documented
├── docker-compose.yml               # ClearML creds via env, repo bind-mounted
├── .dockerignore
├── Makefile                         # every reported number = one target
│
├── mlops/                           # MLOps pipeline (the system contribution)
│   ├── __init__.py
│   ├── requirements.txt             # pinned versions (==), no ranges
│   ├── clearml_tracking.py          # init / log_epoch / register_model helpers
│   ├── train_segformer.py           # reproducible ClearML-tracked training
│   ├── calibrate_ambiguity_band.py  # Figure A + empirical band derivation
│   ├── benchmark_cascade.py         # Figure B + C + system_comparison.md
│   ├── cascaded_inference_pipeline.py  # ClearML Pipeline DAG (production cascade)
│   ├── calibration.json             # generated
│   ├── figures/                     # generated
│   └── results/                     # generated
│
├── app/                             # Streamlit demo (deployed)
│   ├── app.py
│   ├── inference.py
│   ├── requirements.txt
│   └── sample_data/                 # 12 chips bundled in repo
│       ├── S1/  Labels/  splits/
│
├── training_notebooks/
│   ├── run_full_pipeline_colab.ipynb     # one-stop Colab reproduction
│   ├── segformer_training_colab_result.ipynb  # frozen training output (legacy)
│   ├── model_training_colab_result.ipynb      # U-Net frozen output (legacy)
│   ├── ssl4eo_sar_vit_training.ipynb          # SAR-pretrained ViT (not used in final)
│   └── fixed_db_threshold.ipynb               # best classical baseline
│
├── experiments/                     # 11 classical baselines (EDA-driven)
│   ├── image_processing_baselines.ipynb
│   └── remote_sensing_baselines.ipynb
│
├── EDA/                             # exploratory data analysis
│   ├── eda.ipynb
│   ├── README.md
│   └── assets/
│
└── report/
    ├── final_report.md              # the submission
    └── literature_survey.md         # full Related Work
```

---

## Implementation tools

| Category | Tool | Purpose |
|---|---|---|
| Language | Python 3.10 | All scripts |
| Deep learning | PyTorch 2.3.0, Transformers 4.40.2 | SegFormer MiT-B2 |
| Geospatial | rasterio 1.3.10 | SAR GeoTIFF I/O |
| Image processing | scikit-image 0.22.0 | Otsu for bimodality routing |
| **MLOps** | **ClearML 1.16.2** | **Experiments, Datasets, Pipelines, Models** |
| Containers | Docker + docker-compose | Reproducible deployment |
| Build | GNU Make | Single-command targets |
| App layer | Streamlit, FastAPI | Live demo + inference endpoint |
| Cloud | GCS, HF Hub, ClearML SaaS, Streamlit Cloud | Dataset, models, pipelines, UI |

Full pinned list in [`mlops/requirements.txt`](mlops/requirements.txt).

---

## Group members and contributions

| Member | Contribution |
|---|---|
| **Ensoo Suk** | U-Net (ResNet-34) baseline implementation and training; U-Net vs. SegFormer comparison including the Bolivia generalization-gap analysis; empirical ambiguity-band calibration (`mlops/calibrate_ambiguity_band.py`) and Figure A; Streamlit demo application and curated sample-data set |
| **FNU Hardik** | EDA; 11 classical and remote-sensing baselines (Otsu, K-Means, NDFI, cross-pol ratio, Lee filter, Kittler–Illingworth, etc.); SegFormer MiT-B2 training (`mlops/train_segformer.py`); **distribution-aware routing signal design** (Fisher discriminant + Otsu alignment); cascade benchmark (`mlops/benchmark_cascade.py`) producing Figures B and C; literature survey and final report |
| **Muazuddin Syed** | Project setup, Sen1Floods11 dataset acquisition and split organization; ClearML Pipeline DAG (`mlops/cascaded_inference_pipeline.py`) wiring the cascade as queue-able components with full lineage; Docker image, `docker-compose.yml`, `Makefile`, and `REPRODUCING.md` reproducibility scaffolding; system-axis comparison against operational SOTA systems (Copernicus EMS, GFM, HydroSAR, Cloud-to-Street, Google Flood Hub) |

---

## Dataset

Sen1Floods11 contains 446 hand-labeled chips across 11 countries:
**Train:** 252 | **Val:** 89 | **Test:** 90 (10 countries) | **Bolivia:** 15 (held-out for OOD generalization).

Labels: `1` = flood, `0` = non-flood, `−1` = invalid (masked during training and evaluation).

EDA writeup with plots: [`EDA/README.md`](EDA/README.md). Key findings:
- Severe class imbalance (flood ≈ 9 % of valid pixels)
- VV separation flood ↔ land = 7.22 dB; VH separation = 8.63 dB
- Otsu auto-labels are unreliable; HandLabeled chips only

---

## Live demo

[Try it](https://hardikkamboj-sen1floods11-segmentation-appapp-jni2fg.streamlit.app/) —
select any of 12 curated chips (or upload your own `_S1Hand.tif`) and
compare classical, U-Net, and SegFormer side-by-side with per-model
error maps. Runs locally with:

```bash
cd app && pip install -r requirements.txt && streamlit run app.py
```

---

## License

Code: MIT. Sen1Floods11 dataset: see
[the upstream license](https://github.com/cloudtostreet/Sen1Floods11).
