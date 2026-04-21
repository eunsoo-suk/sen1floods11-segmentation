"""
app.py — Streamlit app for Sen1Floods11 flood segmentation demo.

Run with:
    cd app && streamlit run app.py
"""

import io
import tempfile
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inference import (
    VV_MEAN, VV_STD, VH_MEAN, VH_STD, THRESHOLD_DB,
    load_chip,
    predict_ip, load_unet, load_segformer,
    predict_unet, predict_segformer,
    compute_metrics,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Flood Segmentation Demo",
    page_icon  = "🌊",
    layout     = "wide",
)

# ── Paths ──────────────────────────────────────────────────────────────────
APP_DIR    = Path(__file__).parent
SAMPLE_S1  = APP_DIR / "sample_data" / "S1"
SAMPLE_LBL = APP_DIR / "sample_data" / "Labels"

SAMPLE_CHIPS = sorted(
    f.stem.replace("_S1Hand", "")
    for f in SAMPLE_S1.glob("*_S1Hand.tif")
)

# Labels for the chip dropdown — show country + performance hint
CHIP_LABELS = {
    "Sri-Lanka_534068": "Sri Lanka #534068  ✦ best (IoU ≈ 0.99)",
    "USA_905409":       "USA #905409  ✦ best (IoU ≈ 0.94)",
    "Paraguay_34417":   "Paraguay #34417  ✦ SegFormer perfect (IoU = 1.00)",
    "Nigeria_812045":   "Nigeria #812045  ✦ strong (IoU ≈ 0.93)",
    "India_900498":     "India #900498  · average",
    "Pakistan_849790":  "Pakistan #849790  · average",
    "Somalia_699062":   "Somalia #699062  · average",
    "Spain_6860600":    "Spain #6860600  · average",
    "Mekong_333434":    "Mekong #333434  · average",
    "Ghana_97059":      "Ghana #97059  ✗ worst (IoU = 0.00)",
    "Ghana_53713":      "Ghana #53713  ✗ worst (IoU = 0.00)",
    "Ghana_83483":      "Ghana #83483  ✗ worst (IoU = 0.00)",
}


# ── Visualisation helpers ──────────────────────────────────────────────────

def stretch(arr):
    """2–98 percentile contrast stretch to [0, 1]."""
    lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


def mask_to_rgb(mask, valid_mask=None):
    """Binary mask → RGB: blue=flood, white=non-flood, grey=invalid."""
    rgb = np.ones((*mask.shape, 3), dtype=np.float32) * 0.95
    rgb[mask == 1] = [0.12, 0.47, 0.71]
    if valid_mask is not None:
        rgb[valid_mask == 0] = [0.50, 0.50, 0.50]
    return rgb


def diff_to_rgb(pred, label, valid_mask=None):
    """Difference map: white=correct, red=FP, green=FN, grey=invalid."""
    rgb = np.ones((*pred.shape, 3), dtype=np.float32) * 0.95
    if valid_mask is not None:
        rgb[valid_mask == 0]                                 = [0.50, 0.50, 0.50]
        rgb[(valid_mask == 1) & (pred == 1) & (label == 0)] = [0.90, 0.30, 0.30]
        rgb[(valid_mask == 1) & (pred == 0) & (label == 1)] = [0.20, 0.70, 0.20]
    else:
        rgb[(pred == 1) & (label == 0)] = [0.90, 0.30, 0.30]
        rgb[(pred == 0) & (label == 1)] = [0.20, 0.70, 0.20]
    return rgb


def render_grayscale(arr, title, caption=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(stretch(arr), cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    if caption:
        st.caption(caption)


def render_mask(arr, title, valid_mask=None, caption=None, legend=True):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask_to_rgb(arr, valid_mask), interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    if legend:
        patches = [
            mpatches.Patch(color=[0.12, 0.47, 0.71], label="Flood"),
            mpatches.Patch(color=[0.95, 0.95, 0.95], label="Non-flood"),
            mpatches.Patch(color=[0.50, 0.50, 0.50], label="Invalid"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.8)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    if caption:
        st.caption(caption)


def render_diff(pred, label, valid_mask, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(diff_to_rgb(pred, label, valid_mask), interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    patches = [
        mpatches.Patch(color=[0.95, 0.95, 0.95], label="Correct"),
        mpatches.Patch(color=[0.90, 0.30, 0.30], label="False Positive"),
        mpatches.Patch(color=[0.20, 0.70, 0.20], label="False Negative"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.8)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def iou_color(iou):
    """Return a color string based on IoU value."""
    if iou >= 0.75:
        return "green"
    if iou >= 0.50:
        return "orange"
    return "red"


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
        "Flood_-_FEMA_-_45838.jpg/320px-Flood_-_FEMA_-_45838.jpg",
        use_container_width=True,
    )
    st.title("🌊 Flood Segmentation")
    st.caption("Sen1Floods11 · Sentinel-1 SAR")
    st.divider()

    # ── Model weights ──────────────────────────────────────────────────────
    st.subheader("⚙️ Model Weights")
    hf_repo = st.text_input(
        "Hugging Face Repo ID",
        value="YOUR_USERNAME/sen1floods11-models",
        help="The HuggingFace repo where unet_flood_best.pt and "
             "segformer_flood_best.pt are stored.",
    )

    # ── Models to run ──────────────────────────────────────────────────────
    st.subheader("🤖 Models")
    run_ip = st.checkbox("Image Processing  (VV ≤ −13.45 dB)", value=True)
    run_un = st.checkbox("U-Net  (ResNet-34 backbone)",         value=True)
    run_sf = st.checkbox("SegFormer  (MiT-B2)",                 value=True)

    st.divider()

    # ── Image source ───────────────────────────────────────────────────────
    st.subheader("🛰️ Select Image")
    source = st.radio("Source", ["Sample chips", "Upload your own"], label_visibility="collapsed")

    if source == "Sample chips":
        selected_chip = st.selectbox(
            "Chip",
            SAMPLE_CHIPS,
            format_func=lambda c: CHIP_LABELS.get(c, c),
        )
        s1_path    = SAMPLE_S1  / f"{selected_chip}_S1Hand.tif"
        label_path = SAMPLE_LBL / f"{selected_chip}_LabelHand.tif"
        label_path = label_path if label_path.exists() else None
        chip_name  = selected_chip

    else:
        st.info(
            "Upload a 2-band GeoTIFF (VV, VH) from Sen1Floods11's S1Hand folder. "
            "Optionally upload the matching LabelHand .tif for metrics."
        )
        s1_file  = st.file_uploader("SAR image  (.tif)", type=["tif", "tiff"])
        lbl_file = st.file_uploader("Label  (.tif) — optional", type=["tif", "tiff"])

        if s1_file is None:
            st.warning("Upload a SAR image to continue.")
            st.stop()

        tmp_s1 = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_s1.write(s1_file.read())
        tmp_s1.flush()
        s1_path    = Path(tmp_s1.name)
        label_path = None

        if lbl_file is not None:
            tmp_lbl = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp_lbl.write(lbl_file.read())
            tmp_lbl.flush()
            label_path = Path(tmp_lbl.name)

        chip_name = s1_file.name.replace("_S1Hand.tif", "")

    st.divider()
    st.caption("CMSC 605 · UMD · 2026")


# ── Load chip ──────────────────────────────────────────────────────────────
with st.spinner("Loading SAR chip…"):
    s1_raw, s1_norm, label, valid_mask = load_chip(s1_path, label_path)

vv_raw, vh_raw = s1_raw[0], s1_raw[1]
H, W = vv_raw.shape

country  = chip_name.split("_")[0]
chip_id  = "_".join(chip_name.split("_")[1:])


# ── Header ─────────────────────────────────────────────────────────────────
st.title(f"🌊 {country}  —  Chip {chip_id}")
st.caption(
    f"Resolution: {H}×{W} px · 10 m/px · 2-band SAR (VV + VH)  |  "
    f"Device: `{'GPU' if str(s1_norm.dtype) else 'CPU'}`"
)

# ── Section 1: SAR Input ───────────────────────────────────────────────────
st.subheader("SAR Input")
col_vv, col_vh, col_gt = st.columns(3)

with col_vv:
    render_grayscale(
        vv_raw, "VV Band (dB)",
        caption=f"Range: [{vv_raw.min():.1f}, {vv_raw.max():.1f}] dB  ·  "
                f"Mean: {vv_raw.mean():.1f} dB",
    )

with col_vh:
    render_grayscale(
        vh_raw, "VH Band (dB)",
        caption=f"Range: [{vh_raw.min():.1f}, {vh_raw.max():.1f}] dB  ·  "
                f"Mean: {vh_raw.mean():.1f} dB",
    )

with col_gt:
    if label is not None:
        flood_pct = (label[valid_mask == 1].mean() * 100) if valid_mask is not None else label.mean() * 100
        render_mask(
            label, "Ground Truth",
            valid_mask = valid_mask,
            caption    = f"Flood coverage: {flood_pct:.1f}% of valid pixels",
        )
    else:
        st.info("No ground-truth label available for this chip.")

st.divider()

# ── Section 2: Run models & show predictions ───────────────────────────────
st.subheader("Model Predictions")

results = {}   # name → (pred, metrics)

# Image processing baseline (no weight download needed)
if run_ip:
    pred_ip = predict_ip(vv_raw)
    results["IP"] = {
        "label":   "Image Processing\n(VV ≤ −13.45 dB)",
        "pred":    pred_ip,
        "metrics": compute_metrics(pred_ip, label, valid_mask),
    }

# U-Net
if run_un:
    if "YOUR_USERNAME" in hf_repo:
        st.warning("⚠️ Set your Hugging Face Repo ID in the sidebar before loading neural models.")
    else:
        try:
            unet_model = load_unet(hf_repo)
            pred_un    = predict_unet(unet_model, s1_norm)
            results["UNet"] = {
                "label":   "U-Net\n(ResNet-34)",
                "pred":    pred_un,
                "metrics": compute_metrics(pred_un, label, valid_mask),
            }
        except Exception as e:
            st.error(f"U-Net failed: {e}")

# SegFormer
if run_sf:
    if "YOUR_USERNAME" in hf_repo:
        pass   # warning already shown above
    else:
        try:
            sf_model = load_segformer(hf_repo)
            pred_sf  = predict_segformer(sf_model, s1_norm)
            results["SegFormer"] = {
                "label":   "SegFormer\n(MiT-B2)",
                "pred":    pred_sf,
                "metrics": compute_metrics(pred_sf, label, valid_mask),
            }
        except Exception as e:
            st.error(f"SegFormer failed: {e}")

if not results:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

# ── Render predictions ─────────────────────────────────────────────────────
cols = st.columns(len(results))
for col, (key, res) in zip(cols, results.items()):
    with col:
        model_label = res["label"]
        pred        = res["pred"]
        metrics     = res["metrics"]

        # Prediction mask
        render_mask(pred, model_label.replace("\n", " "), valid_mask=valid_mask, legend=False)

        # Error map vs ground truth
        if label is not None:
            render_diff(pred, label, valid_mask, "Error map")

        # Metric cards
        if metrics:
            iou = metrics["IoU"]
            st.metric("IoU",       f"{metrics['IoU']:.3f}")
            st.metric("F1",        f"{metrics['F1']:.3f}")
            st.metric("Precision", f"{metrics['Precision']:.3f}")
            st.metric("Recall",    f"{metrics['Recall']:.3f}")
        else:
            st.caption("Upload a label file for metrics.")

st.divider()

# ── Section 3: Metrics summary table ──────────────────────────────────────
metrics_available = [r for r in results.values() if r["metrics"] is not None]
if metrics_available:
    st.subheader("Metrics Summary")
    rows = []
    for key, res in results.items():
        if res["metrics"]:
            rows.append({
                "Model": res["label"].replace("\n", " "),
                **res["metrics"],
            })

    df = pd.DataFrame(rows).set_index("Model")

    # Highlight best value per column in green
    styled = df.style.highlight_max(
        axis=0,
        subset=["IoU", "F1", "Precision", "Recall", "Accuracy"],
        color="#c6efce",
    ).format("{:.4f}")

    st.dataframe(styled, use_container_width=True)

    # Bar chart of IoU
    st.subheader("IoU Comparison")
    iou_data = {r["label"].replace("\n", " "): r["metrics"]["IoU"]
                for r in results.values() if r["metrics"]}
    st.bar_chart(iou_data, color="#1f77b4")
