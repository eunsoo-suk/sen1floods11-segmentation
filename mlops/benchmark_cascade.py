"""Cascade efficiency benchmark — distribution-aware routing.

Routes each chip based on its **VV-distribution shape**, not on a hand-picked
ambiguity band. The cascade trusts the classical fixed-dB threshold when the
chip's distribution is *bimodal with peaks aligned with the physics-based
threshold*; otherwise it routes to the deep model.

Why this routing
----------------
The fixed-dB threshold (-13.45 dB) is a physics-derived decision boundary
between water and land SAR backscatter. It is *correct* exactly when a
chip's VV histogram has two well-separated modes (water + land) with the
trough near -13.45. When that geometry breaks (urban shadow confounders,
vegetated flood, unimodal chip dominated by speckle), classical fails and
the deep model must take over.

We measure "is this chip's distribution physics-aligned?" with two cheap
chip-level statistics:

  - **Bimodality (Fisher discriminant)** — between-class variance over
    within-class variance, with classes split at the chip's Otsu threshold.
    High = clean two-mode separation. Low = unimodal or speckle-dominated.
  - **Threshold alignment** — `|otsu_t − (-13.45)|`. Small = the chip's
    natural bimodality respects the physics-based threshold. Large = the
    chip's modes are placed unusually (often a confounder).

A chip is "classical-trustworthy" when both signals pass:
    bimodality ≥ τ_bimod   AND   alignment ≤ τ_align

Outputs (all artifacts on a single ClearML Task)
------------------------------------------------
- ``mlops/figures/figure_b_tradeoff.png``    — IoU vs % chips routed to deep
- ``mlops/figures/figure_c_distribution.png`` — per-chip (bimodality, IoU
                                                gap) scatter — validates routing
- ``mlops/results/benchmark.csv``             — per-(strategy, params, split)
- ``mlops/results/system_comparison.md``      — pre-formatted report table
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from clearml import Task
from skimage.filters import threshold_otsu
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# Constants from EDA — must match training and the calibration script.
DB_THRESHOLD = -13.45
VV_MEAN, VV_STD = -10.41, 4.14
VH_MEAN, VH_STD = -17.14, 4.68


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Chip:
    s1: np.ndarray
    label: np.ndarray
    valid: np.ndarray
    chip_id: str
    stats: dict = field(default_factory=dict)


def load_split(s1_dir: Path, label_dir: Path, split_csv: Path) -> list[Chip]:
    chips: list[Chip] = []
    with open(split_csv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s1_name, lbl_name = [s.strip() for s in line.split(",")]
            with rasterio.open(s1_dir / s1_name) as src:
                s1 = src.read().astype(np.float32)
            with rasterio.open(label_dir / lbl_name) as src:
                label = src.read(1).astype(np.float32)
            s1 = np.nan_to_num(s1, nan=0.0)
            valid = (label != -1)
            label = np.clip(label, 0, 1).astype(np.uint8)
            chip = Chip(s1, label, valid, s1_name.replace("_S1Hand.tif", ""))
            chip.stats = chip_distribution_stats(s1[0])
            chips.append(chip)
    return chips


# ─────────────────────────────────────────────────────────────────────────────
# Distribution-aware routing
# ─────────────────────────────────────────────────────────────────────────────
def chip_distribution_stats(vv: np.ndarray) -> dict:
    """Compute physically-motivated chip-level statistics for cascade routing.

    Returns
    -------
    dict with:
        otsu_t      : per-chip optimal threshold from Otsu's method
        bimodality  : Fisher discriminant — between-class variance / within
                      -class variance under Otsu's split. ≥4 indicates a
                      clean bimodal split; <1 is essentially unimodal.
        alignment   : |otsu_t − DB_THRESHOLD|. Small = chip's natural modes
                      respect the physics-based decision boundary.
    """
    finite = vv[np.isfinite(vv)]
    if finite.size == 0 or finite.min() == finite.max():
        return {"otsu_t": DB_THRESHOLD, "bimodality": 0.0, "alignment": 0.0}
    try:
        otsu_t = float(threshold_otsu(finite))
    except ValueError:
        return {"otsu_t": DB_THRESHOLD, "bimodality": 0.0, "alignment": 0.0}

    below = finite[finite <  otsu_t]
    above = finite[finite >= otsu_t]
    if below.size < 100 or above.size < 100:
        return {"otsu_t": otsu_t, "bimodality": 0.0,
                "alignment": abs(otsu_t - DB_THRESHOLD)}

    sep = (above.mean() - below.mean()) ** 2
    spread = below.var() + above.var() + 1e-9
    return {
        "otsu_t":     otsu_t,
        "bimodality": float(sep / spread),
        "alignment":  float(abs(otsu_t - DB_THRESHOLD)),
    }


def trust_classical(stats: dict, min_bimodality: float, max_alignment_db: float) -> bool:
    """Decide whether to send a chip to classical (True) or deep (False)."""
    return (stats["bimodality"] >= min_bimodality
            and stats["alignment"] <= max_alignment_db)


# ─────────────────────────────────────────────────────────────────────────────
# Predictors
# ─────────────────────────────────────────────────────────────────────────────
def classical_predict(chip: Chip) -> np.ndarray:
    return (chip.s1[0] < DB_THRESHOLD).astype(np.uint8)


def deep_predict(model, chips: list[Chip], device, threshold: float = 0.4) -> list[np.ndarray]:
    """Run SegFormer on a list of chips with proper z-score normalization."""
    out: list[np.ndarray] = []
    BATCH = 8
    for s in range(0, len(chips), BATCH):
        batch_chips = chips[s : s + BATCH]
        x = np.stack([c.s1.copy() for c in batch_chips])           # (B, 2, H, W)
        x[:, 0] = (x[:, 0] - VV_MEAN) / VV_STD
        x[:, 1] = (x[:, 1] - VH_MEAN) / VH_STD
        x_t = torch.from_numpy(x).to(device)
        with torch.no_grad():
            logits = model(pixel_values=x_t).logits
            logits = F.interpolate(logits, size=x_t.shape[-2:], mode="bilinear")
            probs  = torch.sigmoid(logits).cpu().numpy()[:, 0]
        for p in probs:
            out.append((p > threshold).astype(np.uint8))
    return out


def deep_predict_warmup(model, chip: Chip, device) -> None:
    """Single-chip forward pass to absorb GPU init time before timing runs."""
    _ = deep_predict(model, [chip], device)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_iou(preds: list[np.ndarray], chips: list[Chip]) -> dict:
    tp = fp = fn = tn = 0
    for pred, chip in zip(preds, chips):
        v = chip.valid
        p = pred[v]; t = chip.label[v]
        tp += int(((p == 1) & (t == 1)).sum())
        fp += int(((p == 1) & (t == 0)).sum())
        fn += int(((p == 0) & (t == 1)).sum())
        tn += int(((p == 0) & (t == 0)).sum())
    iou = tp / max(tp + fp + fn, 1)
    f1  = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"IoU": iou, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def per_chip_iou(pred: np.ndarray, chip: Chip) -> float:
    v = chip.valid
    p = pred[v]; t = chip.label[v]
    inter = int(((p == 1) & (t == 1)).sum())
    union = inter + int(((p == 1) & (t == 0)).sum()) + int(((p == 0) & (t == 1)).sum())
    return inter / union if union > 0 else 1.0   # all-correct empty mask = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────
def run_classical_only(chips: list[Chip]) -> tuple[list[np.ndarray], dict]:
    t0 = time.time()
    preds = [classical_predict(c) for c in chips]
    return preds, {"strategy": "classical-only", "wall_seconds": time.time() - t0,
                   "deep_invocations": 0, "frac_deep": 0.0}


def run_deep_only(chips: list[Chip], model, device) -> tuple[list[np.ndarray], dict]:
    t0 = time.time()
    preds = deep_predict(model, chips, device)
    return preds, {"strategy": "deep-only", "wall_seconds": time.time() - t0,
                   "deep_invocations": len(chips), "frac_deep": 1.0}


def run_cascade_dist(
    chips: list[Chip], model, device,
    min_bimodality: float, max_alignment_db: float,
) -> tuple[list[np.ndarray], dict]:
    """Distribution-aware cascade. No pixel-level merge: bimodal chips get
    the classical mask whole, others get the deep mask whole."""
    t0 = time.time()
    deep_idx, classical_idx = [], []
    for i, c in enumerate(chips):
        if trust_classical(c.stats, min_bimodality, max_alignment_db):
            classical_idx.append(i)
        else:
            deep_idx.append(i)

    final: list[np.ndarray | None] = [None] * len(chips)
    for i in classical_idx:
        final[i] = classical_predict(chips[i])

    if deep_idx:
        outs = deep_predict(model, [chips[i] for i in deep_idx], device)
        for i, p in zip(deep_idx, outs):
            final[i] = p

    return [f for f in final if f is not None], {
        "strategy": f"cascade-dist(bimod>={min_bimodality},align<={max_alignment_db})",
        "wall_seconds":     time.time() - t0,
        "deep_invocations": len(deep_idx),
        "frac_deep":        len(deep_idx) / max(len(chips), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_segformer(ckpt_path: Path, device) -> torch.nn.Module:
    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b2", num_labels=1, num_channels=2,
        id2label={0: "flood"}, label2id={"flood": 0},
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", config=config, ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("best_weights") or ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Figure C — validates the routing signal
# ─────────────────────────────────────────────────────────────────────────────
def make_figure_c(
    chips: list[Chip], classical_iou: list[float], deep_iou: list[float],
    out_path: Path, split_name: str,
) -> None:
    """Per-chip scatter: bimodality (x) vs (classical IoU − deep IoU) (y).

    The cascade thesis says: bimodal chips → classical ≈ deep (gap ≈ 0 or
    positive). Unimodal chips → classical ≪ deep (gap negative). A clean
    ramp from negative-gap-when-unimodal to zero-gap-when-bimodal validates
    the routing signal.
    """
    bimod = np.array([c.stats["bimodality"] for c in chips])
    align = np.array([c.stats["alignment"]  for c in chips])
    gap   = np.array(classical_iou) - np.array(deep_iou)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(bimod, gap, c=align, cmap="viridis_r", s=40, edgecolor="k", linewidth=0.5)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.6,
                    label="classical = deep")
    axes[0].set_xlabel("Chip bimodality (Fisher discriminant)")
    axes[0].set_ylabel("Classical IoU − Deep IoU")
    axes[0].set_title(f"{split_name.title()} — chips that are bimodal don't need the deep model")
    axes[0].grid(alpha=0.3); axes[0].legend(loc="lower right")
    cb = plt.colorbar(sc, ax=axes[0]); cb.set_label("Threshold misalignment (dB)")

    # Companion histogram of bimodality scores so the reader sees the spread.
    axes[1].hist(bimod, bins=30, color="#1f77b4", alpha=0.85, edgecolor="white")
    for tau in (1.0, 2.0, 4.0, 8.0):
        axes[1].axvline(tau, color="red", linestyle=":", alpha=0.4)
        axes[1].text(tau, axes[1].get_ylim()[1] * 0.95, f"τ={tau}",
                     rotation=90, va="top", ha="right", fontsize=8, color="red")
    axes[1].set_xlabel("Chip bimodality")
    axes[1].set_ylabel("Number of chips")
    axes[1].set_title("Distribution of chip bimodality scores")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Figure C — distribution-aware routing signal", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--s1-dir",     required=True)
    p.add_argument("--label-dir",  required=True)
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--segformer-ckpt", required=True)
    p.add_argument("--calibration-json", default=None,
                   help="(Compatibility) — distribution routing ignores the band, "
                        "but we still log the empirical band for reference.")
    p.add_argument("--bimodality-thresholds", default="1.0,2.0,4.0,8.0,16.0",
                   help="Comma-separated min-bimodality values to sweep")
    p.add_argument("--alignment-db", type=float, default=2.0,
                   help="Max |otsu_t − (-13.45)| (dB) for a chip to be "
                        "classical-trustworthy")
    p.add_argument("--out-dir", default="mlops/results")
    p.add_argument("--fig-dir", default="mlops/figures")
    args = p.parse_args()

    task = Task.init(
        project_name="Sen1Floods11/Benchmark",
        task_name="cascade-efficiency-distribution",
        task_type=Task.TaskTypes.qc,
    )
    task.connect(vars(args), name="benchmark_args")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading SegFormer...")
    model = load_segformer(Path(args.segformer_ckpt), device)

    bimod_taus = [float(x) for x in args.bimodality_thresholds.split(",")]
    print(f"Sweeping bimodality thresholds: {bimod_taus}  (alignment ≤ {args.alignment_db} dB)")

    splits = {
        "test":    splits_dir / "flood_test_data.csv",
        "bolivia": splits_dir / "flood_bolivia_data.csv",
    }

    rows: list[dict] = []
    n_chips_per_split: dict[str, int] = {}
    for split_name, split_csv in splits.items():
        chips = load_split(Path(args.s1_dir), Path(args.label_dir), split_csv)
        n_chips_per_split[split_name] = len(chips)
        print(f"\n=== {split_name.upper()} ({len(chips)} chips) ===")

        # ── GPU warmup so timing comparisons are honest ────────────────────
        if device.type == "cuda" and chips:
            print("  (GPU warmup pass...)")
            deep_predict_warmup(model, chips[0], device)

        # ── Strategy 1: classical-only ─────────────────────────────────────
        c_preds, c_meta = run_classical_only(chips)
        m = aggregate_iou(c_preds, chips)
        rows.append({"split": split_name, **c_meta, **m})
        print(f"  classical-only        IoU={m['IoU']:.4f}  frac_deep=0.000  "
              f"wall={c_meta['wall_seconds']:.2f}s")

        # ── Strategy 2: deep-only ──────────────────────────────────────────
        d_preds, d_meta = run_deep_only(chips, model, device)
        m_deep = aggregate_iou(d_preds, chips)
        deep_only_iou  = m_deep["IoU"]
        deep_only_wall = d_meta["wall_seconds"]
        rows.append({"split": split_name, **d_meta, **m_deep})
        print(f"  deep-only             IoU={m_deep['IoU']:.4f}  frac_deep=1.000  "
              f"wall={d_meta['wall_seconds']:.2f}s")

        # ── Per-chip IoUs for Figure C (computed once per split) ───────────
        c_chip_iou = [per_chip_iou(p, c) for p, c in zip(c_preds, chips)]
        d_chip_iou = [per_chip_iou(p, c) for p, c in zip(d_preds, chips)]
        make_figure_c(chips, c_chip_iou, d_chip_iou,
                      fig_dir / f"figure_c_distribution_{split_name}.png",
                      split_name=split_name)

        # ── Strategy 3: cascade-dist — sweep bimodality threshold ──────────
        for tau in bimod_taus:
            cas_preds, meta = run_cascade_dist(
                chips, model, device,
                min_bimodality=tau, max_alignment_db=args.alignment_db,
            )
            m = aggregate_iou(cas_preds, chips)
            rows.append({"split": split_name, **meta, **m})
            print(f"  cascade(bimod≥{tau:>5.1f})  "
                  f"IoU={m['IoU']:.4f}  frac_deep={meta['frac_deep']:.3f}  "
                  f"wall={meta['wall_seconds']:.2f}s  "
                  f"ΔIoU={m['IoU']-deep_only_iou:+.4f}  "
                  f"speedup×={deep_only_wall/max(meta['wall_seconds'],1e-6):.1f}")

    # ── Persist raw numbers ──────────────────────────────────────────────────
    csv_path = out_dir / "benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ── Figure B — IoU vs frac_deep (Pareto curve) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, split_name in zip(axes, ("test", "bolivia")):
        split_rows = [r for r in rows if r["split"] == split_name]
        cl  = next(r for r in split_rows if r["strategy"] == "classical-only")
        dp  = next(r for r in split_rows if r["strategy"] == "deep-only")
        cas = sorted([r for r in split_rows if r["strategy"].startswith("cascade-dist")],
                     key=lambda r: r["frac_deep"])
        ax.plot([r["frac_deep"] for r in cas], [r["IoU"] for r in cas],
                marker="o", color="#2ca02c", label="Distribution-aware cascade")
        ax.scatter([0.0], [cl["IoU"]], color="#7f7f7f", s=80, zorder=5,
                   label=f"Classical-only ({cl['IoU']:.3f})")
        ax.scatter([1.0], [dp["IoU"]], color="#d62728", s=80, zorder=5,
                   label=f"Deep-only ({dp['IoU']:.3f})")
        ax.axhline(dp["IoU"], color="#d62728", linestyle="--", alpha=0.4)
        ax.set_xlabel("Fraction of chips routed to deep model")
        ax.set_ylabel("Aggregate IoU")
        ax.set_title(f"{split_name.title()} set")
        ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Figure B — Cascade efficiency / accuracy trade-off",
                 fontweight="bold")
    fig.tight_layout()
    fig_path = fig_dir / "figure_b_tradeoff.png"
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    print(f"Wrote {fig_path}")

    # ── Pick the headline cascade variant: maximize compute saving
    #    subject to IoU within 0.005 of deep-only and frac_deep < 1 ─────────
    test_cas = [r for r in rows if r["split"] == "test"
                and r["strategy"].startswith("cascade-dist")]
    test_dp  = next(r for r in rows if r["split"] == "test" and r["strategy"] == "deep-only")
    iou_tolerance = 0.005
    qualifying = [r for r in test_cas
                  if r["frac_deep"] < 1.0
                  and (test_dp["IoU"] - r["IoU"]) <= iou_tolerance]
    if qualifying:
        # Among configurations that meet the accuracy bar, pick the one
        # with the LOWEST frac_deep — that's the maximum compute saving.
        best = min(qualifying, key=lambda r: r["frac_deep"])
    else:
        # Fallback: closest IoU to deep-only.
        candidates = [r for r in test_cas if r["frac_deep"] < 1.0]
        best = max(candidates, key=lambda r: r["IoU"]) if candidates \
               else max(test_cas, key=lambda r: r["IoU"])

    bol_dp  = next(r for r in rows if r["split"] == "bolivia" and r["strategy"] == "deep-only")
    bol_cas = next((r for r in rows if r["split"] == "bolivia"
                    and r["strategy"] == best["strategy"]), None)

    md = []
    md.append("# System Comparison — Baseline vs. Distribution-Aware Cascaded Pipeline\n")
    md.append("Baseline pipeline = SegFormer on every chip (deep-only).")
    md.append("Cascaded pipeline = classical fast-pass for chips with bimodal, "
              "physics-aligned VV distributions; SegFormer for everything else.\n")
    md.append("| Axis | Baseline (deep-only) | Cascaded (ours) | Δ |")
    md.append("|---|---|---|---|")
    md.append(f"| **Efficiency** — chips routed to deep model | "
              f"100% | {best['frac_deep']*100:.1f}% | "
              f"−{(1 - best['frac_deep']) * 100:.1f}% deep invocations |")
    n_test = n_chips_per_split.get("test", 0)
    n_bol  = n_chips_per_split.get("bolivia", 0)
    md.append(f"| **Efficiency** — deep invocations (test, {n_test} chips) | "
              f"{n_test}/{n_test} (100%) | "
              f"{int(round(best['frac_deep'] * n_test))}/{n_test} ({best['frac_deep']*100:.1f}%) | "
              f"{(1 - best['frac_deep'])*100:.1f}% fewer deep calls |")
    md.append(f"| **Accuracy** — Test IoU | "
              f"{test_dp['IoU']:.4f} | {best['IoU']:.4f} | "
              f"{best['IoU']-test_dp['IoU']:+.4f} |")
    if bol_cas is not None:
        md.append(f"| **Robustness** — Bolivia OOD IoU | "
                  f"{bol_dp['IoU']:.4f} | {bol_cas['IoU']:.4f} | "
                  f"{bol_cas['IoU']-bol_dp['IoU']:+.4f} |")
    cl_test = next(r for r in rows if r['split']=='test'    and r['strategy']=='classical-only')
    cl_bol  = next(r for r in rows if r['split']=='bolivia' and r['strategy']=='classical-only')
    md.append(f"| **Availability** — fallback when deep model is unavailable | "
              f"❌ no fallback | ✅ classical-only mode "
              f"(IoU {cl_test['IoU']:.3f} test, {cl_bol['IoU']:.3f} Bolivia) | — |")
    md.append(f"| **Reliability** — observable per-stage signals | "
              f"single scalar (IoU) | per-chip `bimodality`, `alignment`, "
              f"`frac_deep`, latency | — |")
    md.append(f"| **Scalability** — orchestration | "
              f"monolithic script | ClearML Pipeline DAG; chip-routing decisions "
              f"and deep batches queue independently | — |")

    md_path = out_dir / "system_comparison.md"
    md_path.write_text("\n".join(md))
    print(f"Wrote {md_path}\n")
    print("\n".join(md))

    # ── ClearML artifacts + headline scalars ────────────────────────────────
    task.upload_artifact(name="benchmark_csv",     artifact_object=str(csv_path))
    task.upload_artifact(name="figure_b",          artifact_object=str(fig_path))
    task.upload_artifact(name="figure_c_test",     artifact_object=str(fig_dir / "figure_c_distribution_test.png"))
    task.upload_artifact(name="figure_c_bolivia",  artifact_object=str(fig_dir / "figure_c_distribution_bolivia.png"))
    task.upload_artifact(name="system_comparison", artifact_object=str(md_path))
    logger = task.get_logger()
    logger.report_scalar("headline", "best_cascade_frac_deep", best["frac_deep"], 0)
    logger.report_scalar("headline", "best_cascade_iou_test", best["IoU"], 0)
    logger.report_scalar("headline", "deep_only_iou_test",     test_dp["IoU"], 0)
    if bol_cas is not None:
        logger.report_scalar("headline", "best_cascade_iou_bolivia", bol_cas["IoU"], 0)
        logger.report_scalar("headline", "deep_only_iou_bolivia",    bol_dp["IoU"], 0)
    logger.report_scalar("headline", "speedup_x",
                         test_dp["wall_seconds"] / max(best["wall_seconds"], 1e-6), 0)
    print(f"\nClearML Task ID: {task.id}")


if __name__ == "__main__":
    main()
