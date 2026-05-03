"""Cascaded inference pipeline (Novelty 1) — ClearML Pipelines.

A ClearML pipeline that runs a *cheap* classical SAR baseline (fixed-dB threshold
on VV) on every pixel first, identifies pixels where the classical method is
uncertain (backscatter within an ambiguity band around the threshold), and only
invokes the deep model on those pixels. Final mask = classical-confident
predictions ∪ deep-model predictions on the uncertain band.

Why this is novel
-----------------
Standard flood-segmentation MLOps treats one model as a monolith. Our EDA
showed the fixed-dB threshold (-13.45 dB VV) already achieves 0.62 IoU on the
Bolivia held-out set — i.e. it gets most pixels right cheaply. The hard pixels
sit in a narrow band around the threshold. By gating expensive GPU inference
behind a cheap classical filter, we (a) reduce wall-clock latency on full
SAR scenes and (b) preserve deep-model accuracy on the genuinely ambiguous
regions.

Each pipeline step is a ClearML Task — fully reproducible, lineage-tracked,
and re-runnable. Run it via:

    python mlops/cascaded_inference_pipeline.py \\
        --scene /path/to/S1Hand.tif \\
        --model-task-id <clearml-task-id-of-trained-segformer>

or from the ClearML web UI by cloning the registered pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from clearml import PipelineDecorator, Task
from clearml.automation.controller import PipelineController


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step 1: load and tile the SAR scene
# ─────────────────────────────────────────────────────────────────────────────
@PipelineDecorator.component(
    return_values=["tiles", "tile_coords", "scene_meta"],
    cache=True,
    task_type=Task.TaskTypes.data_processing,
    packages=["rasterio", "numpy"],
)
def load_and_tile(scene_path: str, tile_size: int = 512, overlap: int = 128):
    """Read a Sentinel-1 GeoTIFF and slide a (tile_size × tile_size) window
    over it with `overlap` pixels of stride padding. Returns the tile stack,
    the (row, col) origins, and scene-level metadata for stitching.
    """
    import rasterio

    with rasterio.open(scene_path) as src:
        scene = src.read().astype(np.float32)               # (2, H, W)
        scene_meta = {
            "height": src.height,
            "width":  src.width,
            "transform": list(src.transform),
            "crs": str(src.crs),
        }

    _, H, W = scene.shape
    stride = tile_size - overlap
    tiles, coords = [], []
    for r in range(0, max(1, H - tile_size + 1), stride):
        for c in range(0, max(1, W - tile_size + 1), stride):
            r2, c2 = min(r + tile_size, H), min(c + tile_size, W)
            tile = np.zeros((2, tile_size, tile_size), dtype=np.float32)
            tile[:, : r2 - r, : c2 - c] = scene[:, r:r2, c:c2]
            tiles.append(tile)
            coords.append((r, c))

    return np.stack(tiles), coords, scene_meta


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step 2: classical fast-pass — distribution-aware chip routing
# ─────────────────────────────────────────────────────────────────────────────
@PipelineDecorator.component(
    return_values=["classical_mask", "route_to_deep"],
    cache=True,
    task_type=Task.TaskTypes.inference,
    packages=["numpy", "scikit-image", "clearml"],
)
def classical_fastpass(
    tiles: np.ndarray,
    threshold_db: float = -13.45,
    min_bimodality: float = 4.0,
    max_alignment_db: float = 2.0,
    calibration_task_id: str | None = None,
    # Legacy parameters retained for CLI compatibility, no-op here.
    ambiguity_band_db: float = 1.5,
    use_otsu_disagreement: bool = True,
):
    """Distribution-aware chip routing.

    For each tile, decide whether the classical fixed-dB threshold can be
    *trusted* on its own (chip's VV distribution is bimodal and aligned
    with the physics-based threshold) or whether the deep model needs to
    take over. Two cheap chip-level statistics drive the decision:

      - **Bimodality** (Fisher discriminant — between/within-class variance
        under Otsu's split). High = clean two-mode water/land separation.
      - **Threshold alignment** (|otsu_t − threshold_db|). Small = the
        chip's natural decision boundary respects the physics-based one.

    Returns
    -------
    classical_mask : (N, H, W) uint8
        Per-pixel classical predictions for *every* tile (cheap, kept for
        all chips so the deep_refinement step can either trust or override).
    route_to_deep  : (N,) bool
        Per-tile routing decision. True → tile must be re-predicted by the
        deep model in step 3 (the classical mask is overwritten wholesale).
    """
    from skimage.filters import threshold_otsu
    from clearml import Task as _Task

    vv = tiles[:, 0]                                          # (N, H, W) in dB
    classical_mask = (vv < threshold_db).astype(np.uint8)

    # Calibration JSON is informational only under distribution-aware routing.
    if calibration_task_id:
        cal_task = _Task.get_task(task_id=calibration_task_id)
        cal_path = cal_task.artifacts["calibration"].get_local_copy()
        with open(cal_path) as f:
            cal = json.load(f)
        print(f"(Reference only) empirical ambiguity band from calibration: "
              f"±{float(cal['band_edge_db']):.2f} dB")

    # ── Compute chip-level routing statistics ──────────────────────────────
    bimod_arr = np.zeros(vv.shape[0], dtype=np.float32)
    align_arr = np.zeros(vv.shape[0], dtype=np.float32)
    for i in range(vv.shape[0]):
        finite = vv[i][np.isfinite(vv[i])]
        if finite.size < 200 or finite.min() == finite.max():
            continue
        try:
            otsu_t = float(threshold_otsu(finite))
        except ValueError:
            continue
        below = finite[finite <  otsu_t]
        above = finite[finite >= otsu_t]
        if below.size < 100 or above.size < 100:
            align_arr[i] = abs(otsu_t - threshold_db); continue
        sep = (above.mean() - below.mean()) ** 2
        spread = below.var() + above.var() + 1e-9
        bimod_arr[i] = float(sep / spread)
        align_arr[i] = abs(otsu_t - threshold_db)

    trust_classical = (bimod_arr >= min_bimodality) & (align_arr <= max_alignment_db)
    route_to_deep   = ~trust_classical

    # ── Log routing breakdown to ClearML ────────────────────────────────────
    task = Task.current_task()
    if task is not None:
        logger = task.get_logger()
        logger.report_scalar("cascade", "frac_to_deep",       float(route_to_deep.mean()),    0)
        logger.report_scalar("cascade", "median_bimodality",  float(np.median(bimod_arr)),    0)
        logger.report_scalar("cascade", "median_alignment_db", float(np.median(align_arr)),   0)
        logger.report_scalar("cascade", "min_bimodality",     float(min_bimodality),          0)
        logger.report_scalar("cascade", "max_alignment_db",   float(max_alignment_db),        0)

    return classical_mask, route_to_deep


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step 3: deep-model refinement — invoked only on uncertain regions
# ─────────────────────────────────────────────────────────────────────────────
@PipelineDecorator.component(
    return_values=["refined_mask"],
    cache=False,
    task_type=Task.TaskTypes.inference,
    packages=["torch", "transformers", "numpy"],
)
def deep_refinement(
    tiles: np.ndarray,
    classical_mask: np.ndarray,
    route_to_deep: np.ndarray,
    model_task_id: str,
    threshold: float = 0.4,
):
    """Whole-tile cascade refinement. Tiles with `route_to_deep == True`
    are entirely re-predicted by the deep model and that prediction
    *replaces* the classical mask for those tiles. Tiles with
    `route_to_deep == False` keep the classical prediction unchanged.

    `model_task_id` points to the ClearML Task whose output_model holds the
    SegFormer weights — full lineage from training to inference.
    """
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation

    # ── Pull the model artifact from the ClearML training task ───────────────
    src_task = Task.get_task(task_id=model_task_id)
    weights_path = src_task.models["output"][-1].get_local_copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        weights_path, num_labels=1, num_channels=2, ignore_mismatched_sizes=True,
    ).to(device).eval()

    # SegFormer was trained on z-scored inputs — apply the same normalization
    # here. Constants from EDA, identical to the training notebook.
    VV_MEAN, VV_STD = -10.41, 4.14
    VH_MEAN, VH_STD = -17.14, 4.68

    refined = classical_mask.copy()
    deep_idx = np.where(route_to_deep)[0]
    BATCH = 8
    with torch.no_grad():
        for s in range(0, len(deep_idx), BATCH):
            batch_idx = deep_idx[s : s + BATCH]
            raw = tiles[batch_idx].copy()
            raw[:, 0] = (raw[:, 0] - VV_MEAN) / VV_STD
            raw[:, 1] = (raw[:, 1] - VH_MEAN) / VH_STD
            raw = np.nan_to_num(raw, nan=0.0)
            batch = torch.from_numpy(raw).to(device)
            logits = model(pixel_values=batch).logits
            logits = F.interpolate(logits, size=batch.shape[-2:], mode="bilinear")
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            for k, j in enumerate(batch_idx):
                # Whole-tile replacement: deep mask supersedes classical.
                refined[j] = (probs[k] > threshold).astype(np.uint8)

    # Track how much expensive compute we actually used.
    task = Task.current_task()
    if task is not None:
        task.get_logger().report_scalar(
            title="cascade", series="fraction_tiles_to_deep",
            value=float(route_to_deep.mean()), iteration=0,
        )

    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline step 4: stitch tiles back into a full-scene mask (overlap blending)
# ─────────────────────────────────────────────────────────────────────────────
@PipelineDecorator.component(
    return_values=["scene_mask_path"],
    cache=False,
    task_type=Task.TaskTypes.data_processing,
    packages=["rasterio", "numpy"],
)
def stitch_tiles(
    refined_mask: np.ndarray,
    tile_coords: list,
    scene_meta: dict,
    out_path: str,
    tile_size: int = 512,
):
    """Reconstruct the full-scene flood mask. Where tiles overlap, average
    the per-tile probability via a 2-D Hann window — kills tile-edge seams.
    """
    import rasterio
    from rasterio.transform import Affine

    H, W = scene_meta["height"], scene_meta["width"]
    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    # 2-D Hann window — peaks in the middle of each tile, zeros at the edges.
    hann_1d = np.hanning(tile_size)
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32) + 1e-3

    for mask, (r, c) in zip(refined_mask, tile_coords):
        r2, c2 = min(r + tile_size, H), min(c + tile_size, W)
        h, w = r2 - r, c2 - c
        accum[r:r2, c:c2]  += mask[:h, :w].astype(np.float32) * hann_2d[:h, :w]
        weight[r:r2, c:c2] += hann_2d[:h, :w]

    final = (accum / np.maximum(weight, 1e-6) > 0.5).astype(np.uint8)

    transform = Affine(*scene_meta["transform"][:6])
    with rasterio.open(
        out_path, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="uint8", crs=scene_meta["crs"], transform=transform,
    ) as dst:
        dst.write(final, 1)

    # Register the output as a ClearML artifact so the full lineage
    # (scene → tiles → cascade → mask) is browsable in the ClearML UI.
    task = Task.current_task()
    if task is not None:
        task.upload_artifact(name="scene_flood_mask", artifact_object=out_path)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline definition
# ─────────────────────────────────────────────────────────────────────────────
@PipelineDecorator.pipeline(
    name="sen1floods11-cascaded-inference",
    project="Sen1Floods11/Inference",
    version="0.1.0",
)
def run_pipeline(
    scene_path: str,
    model_task_id: str,
    out_path: str = "/tmp/scene_flood_mask.tif",
    threshold_db: float = -13.45,
    min_bimodality: float = 4.0,
    max_alignment_db: float = 2.0,
    deep_threshold: float = 0.4,
    calibration_task_id: str | None = None,
):
    tiles, tile_coords, scene_meta = load_and_tile(scene_path)
    classical_mask, route_to_deep = classical_fastpass(
        tiles,
        threshold_db=threshold_db,
        min_bimodality=min_bimodality,
        max_alignment_db=max_alignment_db,
        calibration_task_id=calibration_task_id,
    )
    refined_mask = deep_refinement(
        tiles, classical_mask, route_to_deep,
        model_task_id=model_task_id, threshold=deep_threshold,
    )
    scene_mask_path = stitch_tiles(
        refined_mask, tile_coords, scene_meta, out_path=out_path,
    )
    return scene_mask_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to S1 GeoTIFF")
    p.add_argument("--model-task-id", required=True,
                   help="ClearML Task ID of the trained SegFormer model")
    p.add_argument("--out", default="/tmp/scene_flood_mask.tif")
    p.add_argument("--threshold-db", type=float, default=-13.45)
    p.add_argument("--min-bimodality", type=float, default=4.0,
                   help="Min Fisher-discriminant bimodality for a chip to be "
                        "trusted by the classical method (higher = stricter)")
    p.add_argument("--max-alignment-db", type=float, default=2.0,
                   help="Max |otsu_t − threshold_db| (dB) for a chip to be "
                        "trusted by the classical method")
    p.add_argument("--calibration-task-id", default=None,
                   help="ClearML Task ID from calibrate_ambiguity_band.py — "
                        "informational only under distribution-aware routing")
    p.add_argument("--deep-threshold", type=float, default=0.4)
    p.add_argument("--local", action="store_true",
                   help="Run pipeline steps locally instead of on ClearML agents")
    args = p.parse_args()

    if args.local:
        # Steps run in-process — useful for debugging.
        PipelineDecorator.run_locally()
    else:
        # Default: each step is queued and executed by a ClearML Agent.
        PipelineDecorator.set_default_execution_queue("default")

    run_pipeline(
        scene_path=args.scene,
        model_task_id=args.model_task_id,
        out_path=args.out,
        threshold_db=args.threshold_db,
        min_bimodality=args.min_bimodality,
        max_alignment_db=args.max_alignment_db,
        calibration_task_id=args.calibration_task_id,
        deep_threshold=args.deep_threshold,
    )


if __name__ == "__main__":
    main()
