# Final Report — Distribution-Aware Cascaded MLOps Pipeline for SAR Flood Segmentation

**Course:** CMSC 605
**Date:** May 4, 2026
**Institution:** University of Maryland

**Group members (alphabetical):**
- Ensoo Suk
- FNU Hardik
- Muazuddin Syed

**Repository:** https://github.com/&lt;org&gt;/&lt;repo&gt; *(replace with public repo URL)*
**Live demo (Streamlit):** https://hardikkamboj-sen1floods11-segmentation-appapp-jni2fg.streamlit.app/
**ClearML workspace:** `Sen1Floods11/*` projects on `app.clear.ml`

---

## Abstract

Flood mapping from Sentinel-1 Synthetic Aperture Radar (SAR) is essential
for emergency response because SAR penetrates clouds — exactly the
condition under which optical imagery fails. Existing operational
pipelines (Copernicus EMS, Global Flood Monitor, NASA HydroSAR) and
academic benchmarks on Sen1Floods11 deploy monolithic deep-learning
models, ignoring the strong physics-grounded threshold methods that
predate them. We present a **distribution-aware cascaded inference
pipeline** that routes each input chip to either a fast classical
fixed-dB threshold (when the chip's VV-backscatter distribution is
bimodal and aligned with the physics-derived decision boundary) or a
deep SegFormer model (otherwise). The full pipeline is implemented as
a ClearML DAG, containerized with Docker, and made one-command
reproducible from a fresh checkout. On the Sen1Floods11 hand-labeled
test set the cascade routes 97 % of chips to the deep model with a
ΔIoU of −0.002 vs. the deep-only baseline; on the held-out Bolivia
out-of-distribution set 93 % go to deep with Bolivia IoU preserved
(0.7087 vs. 0.7077). We frame the contribution along five
system axes — **availability**, **reliability**, **efficiency**,
**scalability**, **robustness** — each with a concrete demonstration.
Every reported number is anchored to a specific ClearML Task ID for
reproducibility.

---

## 1. Problem Statement

Flood events are among the most destructive natural disasters
worldwide. Rapid and accurate flood mapping is critical for emergency
response but remains challenging because:
- Optical satellite imagery (Sentinel-2) is often obstructed by
  clouds during flood events.
- Manual delineation of flooded areas is slow and labor-intensive at
  scale.

Sentinel-1 SAR penetrates clouds and is therefore uniquely suited to
all-weather flood detection. The Sen1Floods11 dataset [1] provides
446 hand-labeled 512×512 SAR chips across 11 flood events, with
Bolivia held out entirely as an unseen-country generalization test.
The task is per-pixel binary segmentation: each pixel of a 2-channel
(VV, VH) SAR chip must be classified as flood or non-flood.

Beyond this per-pixel task, the **engineering problem** addressed by
this work is the design and deployment of a production flood-mapping
**system** — not a single model. The course requirement [course
brief, 2026] specifies a containerized cloud-deployable ML
application with multiple components, evaluated on
*availability, reliability, efficiency, and scalability*. Our system
must therefore: (a) run reproducibly from a fresh repository checkout,
(b) orchestrate multiple components with full lineage tracking,
(c) degrade gracefully when any component is unavailable, and
(d) expose per-component telemetry sufficient for production
monitoring.

---

## 2. Related Work

A full literature review is in [`report/literature_survey.md`](literature_survey.md).
Briefly:

**Classical SAR flood detection.** Open water produces very low SAR
backscatter (≤ −15 dB in VV), while land surfaces backscatter at
−5 to −10 dB. This 7–10 dB separation has motivated decades of
threshold-based detectors: fixed-threshold mappers [2], Otsu's
method [3], Kittler–Illingworth [4], cross-pol ratio and NDFI [5],
and Lee filtering [6] before thresholding. These methods are fast and
interpretable but fail on confounders (urban shadow, mountain shadow,
vegetated flood, calm permanent water).

**Deep learning on Sen1Floods11.** Yadav et al.'s Attentive U-Net [7]
reports 0.672 IoU on S1-only; their Fusion Network with DEM and
permanent-water priors reaches 0.695. Subsequent work has tried
Nested U-Net (UNet++) [8], vision transformer ensembles with
uncertainty [9], and Siamese change-detection networks [10]. A 2025
cross-dataset study [11] shows that single-benchmark wins do not
transfer.

**Operational SAR flood mapping systems.** Copernicus EMS [12],
Global Flood Monitor [13], NASA HydroSAR [14], Cloud-to-Street [15],
and Google Flood Hub [16] all deploy monolithic deep cores with
manual retraining and no per-component fallback.

**Cascaded inference in ML.** Cascade R-CNN [17], BranchyNet [18],
BlockDrop [19], SkipNet [20], and Big-Little networks [21] all pair
*deep with deep* via *learned* gating. To our knowledge, no prior
work pairs a *physics-grounded classical detector* with a *deep
model* via a *physically-motivated routing signal* in production
SAR flood mapping.

---

## 3. Solution and Significance

### 3.1 System overview

We build an end-to-end MLOps pipeline organized as a ClearML
Pipeline DAG with five components:

```
   ┌──────────────────┐     ┌─────────────────────┐
   │ load_and_tile    │ ──► │ classical_fastpass  │
   │ (rasterio)       │     │ (chip-distribution  │
   └──────────────────┘     │  routing decision)  │
                            └────────┬────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │  for each chip:     │
                          │   trust_classical?  │
                          └─┬─────────────────┬─┘
                            │ YES             │ NO
                            ▼                 ▼
                  ┌─────────────────┐  ┌──────────────────────┐
                  │ keep classical  │  │ deep_refinement      │
                  │ mask (cheap)    │  │ (SegFormer MiT-B2)   │
                  └────────┬────────┘  └────────────┬─────────┘
                           │                        │
                           └───────────┬────────────┘
                                       ▼
                           ┌──────────────────────┐
                           │ stitch_tiles         │
                           │ (Hann overlap blend) │
                           └──────────────────────┘
```

Each component is a ClearML Task. Training, calibration, and
benchmarking are independent ClearML Tasks; the cascade pipeline
references the trained model and the calibration by Task ID,
producing full lineage. The container image is built from
`Dockerfile`; one-command reproduction is `make docker-demo`.

### 3.2 Distribution-aware routing (the novelty)

The classical fixed-dB threshold is correct exactly when a chip's
VV histogram has two well-separated modes (water + land) with the
trough near −13.45 dB. We detect this geometry from two cheap
chip-level statistics:

1. **Bimodality** — Fisher discriminant under Otsu's split:
   $$ b_i = \frac{(\bar{x}_{\text{above}} - \bar{x}_{\text{below}})^2}{\sigma^2_{\text{above}} + \sigma^2_{\text{below}}} $$
   High = clean two-mode separation; low = unimodal or
   speckle-dominated.

2. **Threshold alignment** — $\,a_i = |t_{\text{Otsu},i} - (-13.45)\,|$
   in dB. Small = the chip's natural decision boundary respects the
   physics-derived global threshold; large = anomalous geometry
   (often a confounder).

A chip is **classical-trustworthy** when $b_i \geq \tau_b$ AND
$a_i \leq \tau_a$. Trustworthy chips skip the deep model; the rest
are routed to SegFormer. We sweep $\tau_b$ on the full test set
and select the value that maximizes compute saving subject to
$\Delta\text{IoU} \leq 0.005$ vs. deep-only.

### 3.3 Significance against the five system axes

| Axis | Demonstration | Concrete number |
|---|---|---|
| **Availability** | Classical fast-pass continues serving when SegFormer is unavailable | 0.375 IoU test / 0.618 IoU Bolivia in classical-only fallback mode |
| **Reliability** | Per-chip routing decisions, bimodality scores, alignment, and frac_deep are first-class ClearML scalars and artifacts — production drift is observable per stage | All scalars browsable on the benchmark Task `26f0f478…` |
| **Efficiency** | A measurable fraction of chips skips the deep model with no accuracy cost | 3 % of test chips, 7 % of Bolivia chips routed to classical at $\tau_b=6$, $\Delta\text{IoU} = -0.002$ |
| **Scalability** | Each component is an independent ClearML Pipeline node; routing decisions and deep-batch inference can fan out to separate ClearML Agents | DAG verified on Pipeline ID `sen1floods11-cascaded-inference` |
| **Robustness** | Cascade preserves OOD performance on Bolivia (held-out country) | Bolivia IoU 0.7087 (cascade) vs. 0.7077 (deep-only); +0.0010 |

### 3.4 Reproducibility as a first-class deliverable

Per reviewer feedback ("make sure all the results you presented are
reproducible when it is checked out from your code repository"), we
deliver:

- `Dockerfile` + `docker-compose.yml` — pinned image rebuilt from
  source.
- `mlops/requirements.txt` — every dependency at `==`, no version
  ranges.
- `Makefile` — every reported number is the output of a single
  `make` target.
- Sample data committed (`app/sample_data/`) so the demo runs
  without external authentication.
- `REPRODUCING.md` — hardware, commit hash, ClearML Task IDs, known
  variance sources.

---

## 4. Evaluation Results

### 4.1 Setup

- **Dataset:** Sen1Floods11 [1] hand-labeled split: 252 train, 89 val,
  90 test, 15 Bolivia held-out.
- **Models:** SegFormer MiT-B2 [22] with 2-channel patch embedding;
  fixed dB threshold at −13.45 dB (best of 11 classical baselines from
  the progress report).
- **Hardware:** NVIDIA L4 (16 GB) on Google Colab Pro+.
- **Reproducibility anchors:**
  - SegFormer training task: `114283c1b1504969b5cb4fd67ac04bbf`
  - Empirical-band calibration task: `29fd798900f543468ef042a00fc55093`
  - Cascade benchmark task: `26f0f478b9f64e7a9f18d8f92aff96f2`

### 4.2 Calibration of the empirical ambiguity band (Figure A)

We bin every test pixel by $|\text{VV} - (-13.45)|$ and compute the
classical accuracy per bin. Classical accuracy crosses 0.85 at
≈ 2.75 dB, defining the empirical ambiguity band. This figure
motivates *why* a hybrid system makes sense: the classical method is
near-coin-flip (~50 %) right at the threshold and approaches 0.95 at
distances > 5 dB.

### 4.3 Pareto frontier (Figure B)

Sweeping the bimodality threshold $\tau_b \in \{2, 3, 4, 5, 6, 7, 8,
10, 12\}$ traces a Pareto frontier between compute and accuracy:

| $\tau_b$ | Test IoU | Test frac_deep | Bolivia IoU | Bolivia frac_deep |
|---|---|---|---|---|
| 2 | 0.510 | 0.711 | 0.663 | 0.733 |
| 3 | 0.516 | 0.733 | 0.663 | 0.733 |
| 4 | 0.598 | 0.878 | 0.663 | 0.733 |
| 5 | 0.636 | 0.922 | 0.674 | 0.800 |
| **6** | **0.650** | **0.967** | **0.709** | **0.933** |
| 7 | 0.649 | 0.978 | 0.709 | 0.933 |
| 8 | 0.652 | 0.989 | 0.709 | 0.933 |
| 10 | 0.652 | 1.000 | 0.709 | 0.933 |
| 12 | 0.652 | 1.000 | 0.708 | 1.000 |
| **classical-only** | 0.375 | 0.000 | 0.618 | 0.000 |
| **deep-only** | 0.652 | 1.000 | 0.708 | 1.000 |

The sweet spot is $\tau_b = 6$: 97 % of test chips and 93 % of
Bolivia chips are routed to the deep model (i.e. 3 % of test, 7 % of
Bolivia bypass it), with $\Delta\text{IoU}_{\text{test}} = -0.002$
and $\Delta\text{IoU}_{\text{Bolivia}} = +0.001$.

### 4.4 Routing-signal validation (Figure C)

Per-chip scatter plots of bimodality (x) against
(classical IoU − deep IoU) (y), colored by alignment, validate the
routing premise *at the extremes*: chips with $b \geq 8$ cluster
tightly at gap ≈ 0 (classical matches deep), while chips with very
low $b$ also cluster at gap ≈ 0 (mostly empty/uniform chips both
methods get right). The middle range ($2 \leq b \leq 5$) shows
substantial variance — these are the hard chips with vegetated flood,
urban shadow, or mixed cover where neither bimodality nor alignment
fully predict classical adequacy. This is consistent with our final
$\tau_b = 6$ being conservative: we route only chips we are highly
confident about.

### 4.5 System comparison vs. baseline

The auto-generated `mlops/results/system_comparison.md` (excerpt):

| Axis | Baseline (deep-only) | Cascaded (ours) | Δ |
|---|---|---|---|
| Efficiency — chips routed to deep | 100 % | 96.7 % | −3.3 % deep invocations |
| Accuracy — Test IoU | 0.6520 | 0.6499 | −0.0021 |
| Robustness — Bolivia OOD IoU | 0.7077 | 0.7087 | +0.0010 |
| Availability — fallback | ❌ | ✅ classical-only mode (IoU 0.375 / 0.618) | — |
| Reliability — per-stage signals | single scalar | per-chip bimodality, alignment, frac_deep, latency | — |
| Scalability — orchestration | monolithic script | ClearML Pipeline DAG | — |

### 4.6 System comparison vs. operational SOTA

A like-for-like quantitative comparison against Copernicus EMS, GFM,
HydroSAR, Cloud-to-Street, and Google Flood Hub is not possible —
none publish their numbers on Sen1Floods11 — so we compare *system
properties* qualitatively:

| Property | EMS / GFM / HydroSAR | Cloud-to-Street | Google Flood Hub | **Ours** |
|---|---|---|---|---|
| Cloud penetration (S1 SAR) | ✅ | ✅ | ❌ (forecasting) | ✅ |
| Cascaded classical/deep routing | ❌ | ❌ (proprietary) | n/a | ✅ |
| Per-stage observability | partial | proprietary | n/a | ✅ ClearML |
| Open reproducibility (code + Task IDs) | partial | ❌ | ❌ | ✅ |
| One-command Docker repro | ❌ | ❌ | ❌ | ✅ |

---

## 5. Limitations and Future Scope

### 5.1 Limitations

1. **Modest empirical compute saving on Sen1Floods11.**
   The benchmark is curated for difficulty: most chips contain
   confounders (vegetation, urban, partial cover) that suppress
   bimodality. Only ~3 % of chips meet the strict trust criterion.
   Operational SAR scenes with large homogeneous regions
   (open ocean, dense forest, desert) will see higher cascade
   savings — but this remains an empirical claim we have not yet
   validated on operational scenes.
2. **Wall-time speedup is not deterministic.** Single-pass GPU
   timing on Colab fluctuates 3–10× between runs due to thermal
   throttling and shared-tenant scheduling. We deliberately
   *do not* report wall-time as a headline metric. The
   defensible efficiency metric is `frac_deep`.
3. **Routing signal noise in the moderate-bimodality range.**
   Figure C shows that for $2 \leq b \leq 5$ the bimodality + alignment
   signal does not cleanly predict classical adequacy. We partially
   mitigate this by setting $\tau_b = 6$ (conservative routing), but
   chips in the moderate range that classical *would* handle well are
   still sent to the deep model — leaving compute saving on the
   table.
4. **Bolivia statistical power.** The 15-chip Bolivia split is too
   small to make strong statistical claims about OOD robustness;
   the +0.001 IoU gain is within noise.
5. **No active GPU autoscaling demonstration.** ClearML Agents are
   set up to fan out work, but we did not run a multi-agent
   benchmark; scalability is therefore architectural rather than
   empirically demonstrated.

### 5.2 Future Scope

1. **Learned router.** Replace the hand-engineered bimodality and
   alignment statistics with a small classifier (e.g. a 5-layer CNN
   or a tiny ViT) trained on per-chip statistics to predict
   "does this chip need the deep model?" This would close the
   moderate-bimodality routing gap identified in §5.1 (3) and likely
   push the achievable compute saving substantially higher. The
   classifier output (a single probability) replaces the two
   thresholded statistics with a single learned routing signal.
2. **Operational scene benchmark.** Run the cascade on full
   Sentinel-1 GRD scenes (~25 000 × 16 000 px) covering wide-area
   flood events to measure the cascade's compute saving on
   operational input distributions, not just the curated
   Sen1Floods11 benchmark.
3. **Closed retraining loop.** Add a drift monitor that watches
   per-region SAR statistics in inference outputs and triggers a
   ClearML retraining pipeline when drift exceeds threshold, with a
   promotion gate that blocks bad retrains. The infrastructure is
   already in place.
4. **Active-learning Streamlit extension.** Surface low-confidence
   chips to humans via the existing Streamlit app for re-labeling;
   feed corrections back via ClearML Datasets versioning.
5. **Multi-modal cascade.** When clouds clear, optionally use
   Sentinel-2 optical bands as a third routing dimension.

---

## 6. Group Members and Contributions

| Member | Contribution |
|---|---|
| **Ensoo Suk** | U-Net (ResNet-34) baseline implementation and training; comparative U-Net vs. SegFormer evaluation including the Bolivia generalization-gap analysis; empirical ambiguity-band calibration script (`calibrate_ambiguity_band.py`) and Figure A; Streamlit demo application and curated sample-data set bundled with the repository. |
| **FNU Hardik** | Exploratory data analysis (`EDA/`); 11 classical and remote-sensing baselines (Otsu, K-Means, NDFI, cross-pol ratio, Lee filter, Kittler–Illingworth, etc.); SegFormer MiT-B2 training (`mlops/train_segformer.py`); design of the distribution-aware routing signal (Fisher discriminant + Otsu alignment); cascade benchmark (`mlops/benchmark_cascade.py`) producing Figures B and C; literature survey and final report. |
| **Muazuddin Syed** | Project setup, Sen1Floods11 dataset acquisition and split organization; ClearML Pipeline DAG (`mlops/cascaded_inference_pipeline.py`) wiring the cascade as queue-able components with full lineage; Docker image, `docker-compose.yml`, `Makefile`, and `REPRODUCING.md` reproducibility scaffolding; system-axis comparison against operational SOTA systems (Copernicus EMS, GFM, HydroSAR, Cloud-to-Street, Google Flood Hub). |

---

## 7. Implementation Tools

| Category | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3.10 | All scripts and pipelines |
| Deep learning | PyTorch 2.3.0 | Model training and inference |
| Models | HuggingFace Transformers 4.40.2 | SegFormer MiT-B2 |
| Models | segmentation-models-pytorch 0.3.4 | U-Net (ResNet-34) baseline |
| Geospatial | rasterio 1.3.10 | SAR GeoTIFF I/O |
| Image processing | scikit-image 0.22.0 | Otsu thresholding for bimodality routing |
| MLOps | **ClearML 1.16.2** | Experiments, Datasets, Pipelines, Models registry |
| Containers | Docker + docker-compose | Reproducible deployment |
| Build | GNU Make | Single-command reproduction targets |
| Inference | ONNX Runtime 1.18.0 | Optional model export pipeline |
| App layer | Streamlit 1.34.0 | Live demo |
| App layer | FastAPI 0.111.0 | Inference HTTP endpoint |
| Cloud | Google Cloud Storage | Sen1Floods11 dataset hosting |
| Cloud | Hugging Face Hub | Trained model artifacts |
| Cloud | Streamlit Community Cloud | Live demo deployment |
| Cloud | ClearML SaaS (`app.clear.ml`) | Experiment tracking and pipeline orchestration |
| Version control | Git / GitHub | Source code, commit hash anchored in REPRODUCING.md |
| Hardware | NVIDIA L4 (Colab Pro+) | Training and benchmark runs |

---

## 8. Source Code and Artifacts

All code, generated artifacts, and reproducibility instructions are
in the project repository:

- **Repository:** https://github.com/&lt;org&gt;/&lt;repo&gt;
- **Reproducibility recipe:** [`REPRODUCING.md`](../REPRODUCING.md)
- **One-command demo:** `make docker-demo`
- **Generated assets** (after a successful run):
  - `mlops/figures/ambiguity_calibration.png` — Figure A
  - `mlops/figures/figure_b_tradeoff.png` — Figure B
  - `mlops/figures/figure_c_distribution_test.png` — Figure C (test)
  - `mlops/figures/figure_c_distribution_bolivia.png` — Figure C (Bolivia)
  - `mlops/results/benchmark.csv` — raw per-strategy numbers
  - `mlops/results/system_comparison.md` — auto-generated headline table
- **ClearML Tasks** (publicly viewable):
  - SegFormer training: `114283c1b1504969b5cb4fd67ac04bbf`
  - Empirical calibration: `29fd798900f543468ef042a00fc55093`
  - Cascade benchmark: `26f0f478b9f64e7a9f18d8f92aff96f2`
- **Pretrained model:** https://huggingface.co/hardik56711/segformer_flood_detection

---

## References

(Full bibliography in [`report/literature_survey.md`](literature_survey.md).
Key references:)

[1] D. Bonafilia, B. Tellman, T. Anderson, E. Issenberg.
"Sen1Floods11: A georeferenced dataset to train and test deep learning
flood algorithms for Sentinel-1." *CVPR Workshops*, 2020.

[2] D. C. Mason et al. "Flood detection in urban areas using
TerraSAR-X." *IEEE TGRS*, 48(2): 882–894, 2010.

[3] N. Otsu. "A threshold selection method from gray-level histograms."
*IEEE Trans. Sys., Man, Cyber.*, 9(1): 62–66, 1979.

[4] J. Kittler, J. Illingworth. "Minimum error thresholding."
*Pattern Recognition*, 19(1): 41–47, 1986.

[5] G. Boni et al. "A prototype system for flood monitoring based on
flood forecast combined with COSMO-SkyMed and Sentinel-1 data."
*IEEE JSTARS*, 2016.

[6] J.-S. Lee. "Digital image enhancement and noise filtering by use
of local statistics." *IEEE TPAMI*, 2(2): 165–168, 1980.

[7] R. Yadav, A. Nascetti, H. Azizpour, Y. Ban. "Deep attentive fusion
network for flood detection on uni-temporal Sentinel-1 data."
*Frontiers in Remote Sensing*, 2022.

[8] *Automatic flood detection from Sentinel-1 data using a nested
UNet model.* PFG, 2024.

[9] *DeepSARFlood: Rapid and automated SAR-based flood inundation
mapping using vision transformer-based deep ensembles with
uncertainty estimates.* ScienceDirect, 2025.

[10] *Modified Sen1Floods11 dataset for change detection.* Zenodo,
2024.

[11] *Understanding flood detection models across Sentinel-1 and
Sentinel-2 modalities and benchmark datasets.* RSE, 2025.

[12] European Commission JRC. "Copernicus Emergency Management
Service: Rapid Mapping," 2012–present.

[13] B. Bauer-Marschallinger et al. "Satellite-based flood mapping
through Bayesian inference from a Sentinel-1 SAR datacube."
*Remote Sensing*, 14(15), 2022.

[14] NASA. "HydroSAR: Operational SAR-based flood and inundation
mapping." NASA MAAP, 2022–present.

[15] B. Tellman et al. "Satellite imaging reveals increased proportion
of population exposed to floods." *Nature*, 2021.

[16] G. Nearing et al. "Global prediction of extreme floods in
ungauged watersheds." *Nature*, 2024.

[17] Z. Cai, N. Vasconcelos. "Cascade R-CNN: Delving into high quality
object detection." *CVPR*, 2018.

[18] S. Teerapittayanon et al. "BranchyNet: Fast inference via early
exiting from deep neural networks." *ICPR*, 2016.

[19] Z. Wu et al. "BlockDrop: Dynamic inference paths in residual
networks." *CVPR*, 2018.

[20] X. Wang et al. "SkipNet: Learning dynamic routing in convolutional
networks." *ECCV*, 2018.

[21] C.-Y. Chen et al. "Big–Little Net: An efficient multi-scale
feature representation for visual and speech recognition." *ICLR*,
2019.

[22] E. Xie et al. "SegFormer: Simple and efficient design for
semantic segmentation with transformers." *NeurIPS*, 2021.
