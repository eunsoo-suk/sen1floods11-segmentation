"""
inference.py — model loading + prediction helpers for the Streamlit app.
"""

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from huggingface_hub import hf_hub_download
import streamlit as st

# ── Normalization constants (from EDA) ────────────────────────────────────
VV_MEAN, VV_STD = -10.41, 4.14
VH_MEAN, VH_STD = -17.14, 4.68
THRESHOLD_DB    = -13.45          # fixed VV threshold for image processing baseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ──────────────────────────────────────────────────────────

def load_chip(s1_path, label_path=None):
    """
    Load a SAR chip and optional label.

    Returns:
        s1_raw   : (2, H, W) float32 — raw dB values, for IP baseline + display
        s1_norm  : (2, H, W) float32 — normalised, for UNet / SegFormer input
        label    : (H, W) float32 or None
        valid_mask: (H, W) float32 or None  (1 = valid pixel, 0 = invalid)
    """
    with rasterio.open(s1_path) as src:
        s1_raw = src.read().astype(np.float32)  # (2, H, W)

    label = valid_mask = None
    if label_path is not None:
        with rasterio.open(label_path) as src:
            label_raw = src.read(1).astype(np.float32)
        valid_mask = (label_raw != -1).astype(np.float32)
        label      = np.clip(label_raw, 0, 1)

    # Normalise for model input
    s1_norm    = s1_raw.copy()
    s1_norm[0] = (s1_raw[0] - VV_MEAN) / VV_STD
    s1_norm[1] = (s1_raw[1] - VH_MEAN) / VH_STD
    s1_norm    = np.nan_to_num(s1_norm, nan=0.0)

    return s1_raw, s1_norm, label, valid_mask


# ── Image processing baseline ─────────────────────────────────────────────

def predict_ip(vv_raw):
    """Fixed dB threshold — no model weights needed."""
    return (vv_raw < THRESHOLD_DB).astype(np.float32)


# ── Model loading (cached so weights are downloaded only once) ────────────

@st.cache_resource(show_spinner="Downloading U-Net weights from HuggingFace...")
def load_unet(hf_repo_id):
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,       # weights come from our checkpoint, not ImageNet
        in_channels     = 2,
        classes         = 1,
        activation      = None,
    )
    weights_path = hf_hub_download(repo_id=hf_repo_id, filename="unet_flood_best.pt")
    ckpt  = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


@st.cache_resource(show_spinner="Downloading SegFormer weights from HuggingFace...")
def load_segformer(hf_repo_id):
    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b2",
        num_labels   = 1,
        num_channels = 2,       # VV + VH (overrides ImageNet's 3)
        id2label     = {0: "flood"},
        label2id     = {"flood": 0},
    )
    model = SegformerForSemanticSegmentation(config)
    weights_path = hf_hub_download(repo_id=hf_repo_id, filename="segformer_flood_best.pt")
    ckpt  = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


# ── Inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_unet(model, s1_norm):
    x      = torch.from_numpy(s1_norm).unsqueeze(0).to(DEVICE)  # (1,2,H,W)
    logits = model(x)                                             # (1,1,H,W)
    return (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0].astype(np.float32)


@torch.no_grad()
def predict_segformer(model, s1_norm):
    x       = torch.from_numpy(s1_norm).unsqueeze(0).to(DEVICE)
    outputs = model(pixel_values=x)
    logits  = F.interpolate(
        outputs.logits,
        size          = x.shape[-2:],   # upsample from H/4 → H
        mode          = "bilinear",
        align_corners = False,
    )
    return (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0].astype(np.float32)


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(pred, label, valid_mask):
    """IoU, F1, Precision, Recall, Accuracy — evaluated on valid pixels only."""
    if label is None or valid_mask is None:
        return None
    v  = valid_mask.astype(bool)
    p  = pred[v]
    t  = label[v]
    tp = float(((p == 1) & (t == 1)).sum())
    fp = float(((p == 1) & (t == 0)).sum())
    fn = float(((p == 0) & (t == 1)).sum())
    tn = float(((p == 0) & (t == 0)).sum())
    iou       = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "IoU":       round(iou,       4),
        "F1":        round(f1,        4),
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "Accuracy":  round(accuracy,  4),
    }
