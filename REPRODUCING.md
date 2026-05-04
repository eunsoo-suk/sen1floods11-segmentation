# Reproducing the Reported Results

This document describes the canonical end-to-end reproduction of every number,
figure, and table in the final report. Per the reviewer feedback ("make sure
all the results you presented are reproducible when it is checked out from
your code repository"), running this on a fresh clone should yield numbers
matching the report within tolerances noted below.

---

## Environment

| | |
|---|---|
| Python | 3.10 |
| Pinned dependencies | [`mlops/requirements.txt`](mlops/requirements.txt) — all versions are `==`, no ranges |
| Hardware (reported numbers) | NVIDIA L4, 16 GB VRAM, 16 vCPU (Google Colab Pro+) |
| Hardware (this Docker recipe) | CPU-only is sufficient for inference; GPU optional |
| Operating System | Ubuntu 22.04 (host); the Docker image is `python:3.10-slim` |

---

## Quick start (Docker — recommended)

```bash
git clone <repo-url> && cd group_project

# 1. Set ClearML credentials in your shell (get keys from the ClearML web UI).
export CLEARML_API_ACCESS_KEY=<your-access-key>
export CLEARML_API_SECRET_KEY=<your-secret-key>

# 2. Build the image (one-time, ~3 min).
make docker-build

# 3. Run the end-to-end demo.
make docker-demo
```

This invokes `make demo` inside the container, which runs:
1. `calibrate_ambiguity_band.py` — produces Figure A
2. `benchmark_cascade.py` — produces Figure B + the system comparison table
3. Logs everything to ClearML under projects `Sen1Floods11/Calibration` and
   `Sen1Floods11/Benchmark`.

The sample data (12 chips bundled in `app/sample_data/`) lets the demo run
with no external downloads. **For the full-scale numbers in the report, use
`make download-data` first** (see below).

---

## Quick start (local, no Docker)

```bash
git clone <repo-url> && cd group_project

make setup
clearml-init                  # paste API keys when prompted

make download-data            # full Sen1Floods11 hand-labeled split (~2 GB, gsutil required)
make download-model           # SegFormer checkpoint (~110 MB)
make demo S1_DIR=data/sen1floods11/S1 \
          LABEL_DIR=data/sen1floods11/Labels \
          SPLITS_DIR=data/sen1floods11/splits \
          SEGFORMER_CKPT=checkpoints/segformer_flood_best.pt
```

---

## Per-target reproduction

Every target writes its outputs to a deterministic location and registers
them as ClearML artifacts. Re-running a target is idempotent.

| Target | Outputs (filesystem) | ClearML artifact |
|---|---|---|
| `make calibrate` | `mlops/calibration.json`, `mlops/figures/ambiguity_calibration.png` | Project `Sen1Floods11/Calibration`, task `empirical-ambiguity-band` |
| `make benchmark` | `mlops/results/benchmark.csv`, `mlops/results/system_comparison.md`, `mlops/figures/figure_b_tradeoff.png` | Project `Sen1Floods11/Benchmark`, task `cascade-efficiency` |
| `make pipeline` | `/tmp/scene_flood_mask.tif` | Project `Sen1Floods11/Inference`, pipeline `sen1floods11-cascaded-inference` |

---

## Expected numbers (full Sen1Floods11 hand-labeled split)

These are the headline values cited in the report. Re-running `make benchmark`
on the full dataset should reproduce them within ±0.002 IoU (small variance
from per-chip Otsu thresholding on tied histograms is the only source of
non-determinism).

The benchmark sweeps the bimodality threshold τ; the auto-generated
`system_comparison.md` selects the τ with maximum compute saving subject
to ΔIoU ≤ 0.005 of deep-only. On the full Sen1Floods11 split that lands
at **τ = 6** with `max_alignment_db = 2.0`.

> **Headline:** at τ = 6, the distribution-aware cascade routes
> **97 % of test chips and 93 % of Bolivia chips** to the deep model
> (i.e. **3 % of test, 7 % of Bolivia** are served by the classical
> fast-pass alone). Test IoU drops by **only 0.002** vs. deep-only,
> and Bolivia IoU is **preserved within noise** (+0.001 on 15 chips).

| Split | Strategy | IoU | frac_deep |
|---|---|---|---|
| Test    | classical-only | 0.375 | 0.000 |
| Test    | deep-only      | 0.652 | 1.000 |
| Test    | **cascade τ=6** | **0.650** | **0.967** |
| Bolivia | classical-only | 0.618 | 0.000 |
| Bolivia | deep-only      | 0.708 | 1.000 |
| Bolivia | **cascade τ=6** | **0.709** | **0.933** |

**Wall-time speedups should *not* be cited as a headline result.**
Single-pass GPU timing on Colab fluctuates by 3–10× between runs due
to warm/cold cache and shared-tenant scheduling. The deterministic and
defensible efficiency metric is `frac_deep` — the fraction of chips
that reach the deep model.

The exact numbers (per-τ IoU, frac_deep, per-chip bimodality scores)
are auto-emitted to `mlops/results/benchmark.csv` and the prose summary
to `mlops/results/system_comparison.md` on every run. Copy the
auto-generated table from `system_comparison.md` directly into the
report.

---

## ClearML Task IDs (filled in after first run)

These IDs anchor the report's claims to specific reproducible runs. Update
them after each fresh end-to-end run, before submission.

| Asset | ClearML Task ID | Notes |
|---|---|---|
| SegFormer training | `114283c1b1504969b5cb4fd67ac04bbf` | Project `Sen1Floods11/Training` |
| Empirical calibration | `29fd798900f543468ef042a00fc55093` | Project `Sen1Floods11/Calibration` |
| Cascade benchmark | `26f0f478b9f64e7a9f18d8f92aff96f2` | Project `Sen1Floods11/Benchmark` — sourced for all report tables |
| Cascade pipeline (sample run) | `<task-id>` | Project `Sen1Floods11/Inference` |

Commit hash for the report submission: `<git-rev-parse-HEAD>`

---

## Known sources of variance

- **Otsu disagreement on tied histograms.** When a chip's VV histogram has
  multiple equally-good Otsu thresholds (rare on real SAR data, but possible
  on heavily-saturated chips), the chosen threshold depends on
  `scikit-image` internals. Pinning `scikit-image==0.22.0` makes this
  deterministic across machines.
- **PyTorch CUDA non-determinism.** SegFormer inference uses
  `torch.nn.functional.interpolate(..., mode="bilinear")`, which is not
  bit-exact across CUDA versions. CPU inference (the Docker default) is
  fully deterministic.
- **Sample-data Bolivia split is a placeholder.** The 3 chips in
  `app/sample_data/splits/flood_bolivia_data.csv` are *not* the real
  Sen1Floods11 Bolivia held-out chips — they're disjoint sample chips
  used so `make demo` runs end-to-end without GCS authentication. The
  numbers reported in the paper come from `make download-data` followed
  by `make benchmark` against the real Bolivia split.

---

## Manifest of generated assets

After a successful `make demo` (full data), the following files exist:

```
mlops/
├── calibration.json                # Empirical ambiguity band
├── figures/
│   ├── ambiguity_calibration.png   # Figure A
│   └── figure_b_tradeoff.png       # Figure B
└── results/
    ├── benchmark.csv               # Per-strategy raw numbers
    └── system_comparison.md        # System comparison table (the report's headline)
```

Each is also stored as an artifact on the corresponding ClearML Task.
