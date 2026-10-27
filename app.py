# app.py — RimboAI: Mobile Field Utility for Plant Pathology Diagnostics
# ------------------------------------------------------------------------
import os, io, pickle, uuid, datetime
from typing import Optional, Tuple, List

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from disease_info import get_disease_details

# ========= Lokasi file model & label encoders (Relative Path) =========
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "Model", "NasnetMobile+Augment.keras")
PLANT_PKL_PATH   = os.path.join(BASE_DIR, "Encoder", "plant_encoder.pkl")
DISEASE_PKL_PATH = os.path.join(BASE_DIR, "Encoder", "disease_encoder.pkl")
# =====================================================================

st.set_page_config(
    page_title="RimboAI — Mobile Plant Diagnostics",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Design System: Bright Mobile App Shell with Strict Square Geometry (No AI-Slop, No Rounded Corners)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800;900&display=swap');

    /* STRICT SQUARE GEOMETRY: ALL ELEMENTS MUST HAVE SQUARE CORNERS */
    *, *::before, *::after {
        border-radius: 0px !important;
        box-sizing: border-box;
    }

    html, body, .stApp, p, label, .sec-title, .brand-name, .brand-sub, .step-head, .step-text, .crop-box, .cell-label, .cell-val, .rx-val, .rx-header {
        font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Protect Material Symbols icons used by Streamlit from font override */
    [data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }

    /* Bright light canvas */
    html, body, .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* HIDE STREAMLIT DEFAULT HEADER, TOOLBAR, FORK, GITHUB, AND DOUBLE ARROW */
    header, [data-testid="stHeader"], .stAppHeader {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        padding: 0px !important;
    }
    #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
        width: 0px !important;
    }

    /* HIDE STREAMLIT FOOTER, MANAGE BADGE, AND GITHUB/VIEWER BADGES COMPLETELY */
    footer,
    [data-testid="stFooter"],
    [data-testid="manage-app-button"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    [data-testid="stToolbar"],
    div[class*="viewerBadge"],
    a[class*="viewerBadge"],
    div[class*="ViewerBadge"],
    a[class*="ViewerBadge"],
    [class*="viewerBadge_container"],
    [class*="viewerBadge_link"],
    [class*="viewerBadge_divider"],
    div[class*="ProfileButton"],
    div[class*="manageApp"],
    div[id*="manage-app"],
    a[href*="streamlit.io"],
    a[href*="github.com"],
    div:has(> a[href*="streamlit.io"]),
    div:has(> a[href*="github.com"]),
    div[class*="floatingMenu"],
    div[class*="FloatingMenu"],
    div[class*="floating-menu"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        height: 0px !important;
        width: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
    }

    /* Responsive Mobile Screen Shell */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        max-width: 560px !important;
        margin: 0 auto;
    }

    /* Mobile Top App Bar */
    .mobile-app-bar {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-bottom: 3px solid #16a34a;
        padding: 14px 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-icon {
        font-size: 1.6rem;
        line-height: 1;
    }
    .brand-name {
        font-size: 1.25rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #0f172a;
        line-height: 1.1;
    }
    .brand-name span {
        color: #16a34a;
    }
    .brand-sub {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
    }
    .status-badge {
        background: #dcfce7;
        border: 1px solid #86efac;
        color: #15803d;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        padding: 4px 8px;
        text-transform: uppercase;
    }

    /* Mobile Metric Strip */
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 6px;
        margin-bottom: 16px;
    }
    .strip-item {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #16a34a;
        padding: 8px 4px;
        text-align: center;
    }
    .strip-num {
        font-size: 1.05rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.1;
    }
    .strip-lbl {
        font-size: 0.62rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.03em;
        margin-top: 2px;
    }

    /* Section Header */
    .sec-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 18px 0 10px 0;
    }
    .sec-bar {
        width: 4px;
        height: 16px;
        background: #16a34a;
        flex-shrink: 0;
    }
    .sec-title {
        font-size: 0.88rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #0f172a;
    }

    /* Segmented Selector Style for st.radio */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        gap: 0px;
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        padding: 3px;
        margin-bottom: 12px;
    }
    div[data-testid="stRadio"] label {
        flex: 1;
        text-align: center;
        background: #f8fafc;
        padding: 9px 10px;
        margin: 0 !important;
        border: 1px solid transparent;
        font-weight: 700;
        font-size: 0.82rem;
        color: #334155;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        background: #f1f5f9;
        color: #0f172a;
    }

    /* File Uploader & Camera Box */
    div[data-testid="stFileUploader"] section {
        border: 2px dashed #94a3b8 !important;
        background: #ffffff !important;
        padding: 20px 14px !important;
        text-align: center;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #16a34a !important;
        background: #f0fdf4 !important;
    }
    div[data-testid="stCameraInput"] {
        border: 2px solid #16a34a !important;
        background: #ffffff !important;
        padding: 6px !important;
    }
    div[data-testid="stCameraInput"] button {
        background: #16a34a !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Step Cards on Landing */
    .step-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 18px;
    }
    .step-item {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        padding: 12px 14px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .step-idx {
        background: #0f172a;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 900;
        padding: 3px 7px;
        flex-shrink: 0;
        line-height: 1.2;
    }
    .step-content {
        flex: 1;
    }
    .step-head {
        font-size: 0.88rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .step-text {
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Crop Grid Chips */
    .crop-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 6px;
        margin-bottom: 18px;
    }
    .crop-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 3px solid #16a34a;
        padding: 8px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Mobile Clinical Diagnostic Card */
    .result-card {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        margin: 16px 0;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
    }
    .result-top {
        background: #16a34a;
        color: #ffffff;
        padding: 10px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .result-title {
        font-size: 0.88rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .result-id {
        background: #0f172a;
        color: #ffffff;
        padding: 2px 7px;
        font-size: 0.68rem;
        font-family: monospace;
        font-weight: 700;
    }
    .result-body {
        padding: 12px;
    }
    .result-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .metric-cell {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #16a34a;
        padding: 10px 12px;
    }
    .metric-cell-healthy {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-top: 3px solid #16a34a;
    }
    .metric-cell-disease {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-top: 3px solid #dc2626;
    }
    .cell-label {
        font-size: 0.68rem;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .cell-val {
        font-size: 1.15rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .cell-pill {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 800;
        padding: 2px 6px;
        text-transform: uppercase;
    }
    .pill-green {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .pill-red {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }

    /* Unrecognized / Low Confidence Alert Card */
    .unrec-card {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-top: 4px solid #f59e0b;
        margin: 16px 0;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
    }
    .unrec-top {
        background: #fffbeb;
        border-bottom: 1px solid #fde68a;
        color: #92400e;
        padding: 10px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .unrec-title {
        font-size: 0.88rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #92400e;
    }
    .unrec-body {
        padding: 14px;
    }
    .unrec-notice {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 10px 12px;
        font-size: 0.82rem;
        color: #78350f;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .unrec-guide-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 10px 12px;
        margin-top: 12px;
    }
    .unrec-guide-head {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 6px;
        letter-spacing: 0.04em;
    }

    /* Treatment Prescription Card */
    .rx-card {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-left: 5px solid #16a34a;
        margin: 14px 0;
        padding: 14px;
    }
    .rx-header {
        font-size: 0.88rem;
        font-weight: 900;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1.5px solid #e2e8f0;
    }
    .rx-section {
        margin-bottom: 10px;
    }
    .rx-caption {
        font-size: 0.68rem;
        font-weight: 800;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .rx-val {
        font-size: 0.85rem;
        color: #1e293b;
        line-height: 1.45;
    }
    .rx-action-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 10px 12px;
        margin-top: 10px;
    }
    .rx-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        color: #1e293b;
        line-height: 1.4;
    }
    .rx-marker {
        color: #16a34a;
        font-weight: 900;
        font-size: 0.75rem;
        flex-shrink: 0;
    }

    /* Explainable AI Visual Studio */
    .xai-scale {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        padding: 6px 10px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #475569;
        margin: 8px 0 12px 0;
    }
    .scale-bar {
        height: 8px;
        flex: 1;
        margin: 0 10px;
        border: 1px solid #94a3b8;
        background: linear-gradient(90deg, #1e3a8a 0%, #06b6d4 25%, #eab308 60%, #ef4444 100%);
    }

    /* Tabs Styling: Strict Square */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px !important;
        border-bottom: 2px solid #cbd5e1 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-bottom: none !important;
        padding: 7px 12px !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        color: #334155 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.02em !important;
    }
    .stTabs [aria-selected="true"] {
        background: #16a34a !important;
        color: #ffffff !important;
        border-color: #16a34a !important;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div {
        height: 8px !important;
        background: #e2e8f0 !important;
    }
    div[data-testid="stProgress"] > div > div {
        background: #16a34a !important;
    }

    /* Expander */
    details, [data-testid="stExpander"] {
        border: 1.5px solid #cbd5e1 !important;
        background: #ffffff !important;
    }

    /* Images */
    img, [data-testid="stImage"] img {
        border: 1.5px solid #cbd5e1 !important;
    }

    /* Clean Square Guide Accordion (Zero Broken Ligatures) */
    .guide-accordion {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        margin-bottom: 20px;
    }
    .guide-summary {
        padding: 10px 14px;
        font-size: 0.8rem;
        font-weight: 800;
        color: #0f172a;
        cursor: pointer;
        list-style: none;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border-left: 3px solid #16a34a;
        user-select: none;
    }
    .guide-summary::-webkit-details-marker {
        display: none;
    }
    .guide-chevron {
        font-size: 0.85rem;
        color: #16a34a;
        font-weight: 900;
        transition: transform 0.2s ease;
    }
    .guide-accordion[open] .guide-chevron {
        transform: rotate(180deg);
    }
    .guide-body {
        padding: 12px 14px;
        border-top: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    .guide-row {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        font-size: 0.8rem;
        color: #334155;
        line-height: 1.45;
        margin-bottom: 6px;
    }
    .guide-bullet {
        color: #16a34a;
        font-size: 0.72rem;
        margin-top: 1px;
        flex-shrink: 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 20px 10px 10px 10px;
        color: #64748b;
        font-size: 0.75rem;
        border-top: 1.5px solid #cbd5e1;
        margin-top: 28px;
        line-height: 1.5;
    }
    .app-footer b {
        color: #0f172a;
    }
</style>

<!-- Mobile Top Bar -->
<div class="mobile-app-bar">
    <div class="brand-wrap">
        <span class="brand-icon">🌿</span>
        <div>
            <div class="brand-name">RIMBO<span>AI</span></div>
            <div class="brand-sub">Sistem Diagnosa Patologi Daun</div>
        </div>
    </div>
    <div class="status-badge">● Sistem Siap</div>
</div>

<!-- Mobile Metric Strip -->
<div class="metric-strip">
    <div class="strip-item">
        <div class="strip-num">10</div>
        <div class="strip-lbl">Komoditas</div>
    </div>
    <div class="strip-item">
        <div class="strip-num">24</div>
        <div class="strip-lbl">Penyakit</div>
    </div>
    <div class="strip-item">
        <div class="strip-num">CPU</div>
        <div class="strip-lbl">Inferensi</div>
    </div>
    <div class="strip-item">
        <div class="strip-num">XAI</div>
        <div class="strip-lbl">Eigen-CAM</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Client-Side DOM Script to Continuously Suppress Streamlit Cloud Badges
components.html("""
<script>
function suppressBadges() {
    try {
        const rootDoc = window.parent ? window.parent.document : document;
        const query = [
            'footer',
            '[data-testid="stFooter"]',
            '[data-testid="manage-app-button"]',
            '[data-testid="stStatusWidget"]',
            '[class*="viewerBadge"]',
            '[class*="ViewerBadge"]',
            'a[href*="streamlit.io"]',
            'a[href*="github.com"]'
        ].join(',');
        rootDoc.querySelectorAll(query).forEach(el => {
            el.style.setProperty('display', 'none', 'important');
            el.style.setProperty('visibility', 'hidden', 'important');
            el.style.setProperty('opacity', '0', 'important');
            el.style.setProperty('pointer-events', 'none', 'important');
        });
    } catch (e) {}
}
setInterval(suppressBadges, 500);
document.addEventListener("DOMContentLoaded", suppressBadges);
</script>
""", height=0, width=0)

# ---------- Util Umum ----------
DEFAULT_PLANTS = ["Bean","Betel Leaf","Bitter Gourd","Bottle gourd","Cauliflower",
                  "Cowpea","Cucumber","Eggplant","Radish","Tomato"]
DEFAULT_DISEASES = [
    "Anthracnose", "Anthracnose lesions", "Bacterial wilt", "Black Rot", "Blight", 
    "Downey mildew", "Downy mildew", "Eggplant Cercopora leaf spot", "Eggplant begomovirus", 
    "Eggplant verticillium wilt", "Fusarium wilt", "Leaf_Rot", "Leaf_Spot", "Mosaic Virus", 
    "Mosaic virus", "Radish Black leaf spot", "Radish Downey mildew", "Radish Mosaic virus", 
    "Radish flea beetle", "Rust", "Septoria leaf spot", "Tomato Bacterial spot", 
    "Tomato leaf curl virus", "Tomato spotted wilt"
]

def preprocess_pil(img: Image.Image, img_size=(224,224)) -> np.ndarray:
    img = img.convert("RGB").resize(img_size, Image.BILINEAR)
    return (np.asarray(img).astype("float32") / 255.0)

def ensure_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x);  x = x[None, :] if x.ndim == 1 else x
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def topk_list(probs: np.ndarray, labels: List[str], k: int = 5):
    p = probs.flatten()
    idx = np.argsort(-p)[: min(k, len(p))]
    return [(labels[i] if i < len(labels) else f"class_{i}", float(p[i])) for i in idx]

# ---------- EIGEN-CAM Helper (Class-Agnostic) ----------
def _find_last_conv(model: tf.keras.Model):
    for layer in reversed(model.layers):
        try:
            shp = layer.output_shape
        except Exception:
            continue
        if isinstance(shp, tuple) and len(shp) == 4:
            return layer
    return None

def eigen_cam_from_acts(acts: np.ndarray) -> np.ndarray:
    A = acts[0]  # [H,W,C]
    H, W, C = A.shape
    X = A.reshape(-1, C).astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]
    proj = (X @ pc1).reshape(H, W)

    # Pastikan orientasi positif mengarah pada fitur berenergi tinggi (lesi), bukan background
    if abs(proj.min()) > proj.max():
        proj = -proj

    # Terapkan ReLU: buang aktivasi negatif / background murni (jangan gunakan np.abs)
    heat = np.maximum(proj, 0.0)
    m = heat.max()
    if m > 0:
        heat = heat / m
    return heat

def get_leaf_mask(orig_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(orig_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # Deteksi piksel bukan latar putih/sangat terang dan bukan hitam pekat
    not_white = (gray <= 238).astype(np.float32)
    not_black = (gray >= 15).astype(np.float32)
    mask = not_white * not_black
    # Jika foto kebun alami atau latar belakang kompleks (> 85% terisi), gunakan mask penuh
    if mask.mean() < 0.04:
        return np.ones_like(gray, dtype=np.float32)
    # Dilation halus agar tepi helai daun tidak terpotong kasar
    mask_u8 = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=2)
    return mask_u8.astype(np.float32) / 255.0

def overlay_heatmap_on_img(orig_img: Image.Image, heat01: np.ndarray, alpha: float = 0.50, cutoff: float = 0.15) -> Image.Image:
    orig = np.asarray(orig_img.convert("RGB")).astype(np.float32)
    h, w = orig.shape[:2]

    # 1. Resize heatmap ke ukuran dimensi gambar asli
    hm = cv2.resize(heat01, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Segmentasi Siluet Daun: Isolasi daun dari latar belakang (kertas putih, meja, lantai)
    leaf_mask = get_leaf_mask(orig)
    hm = hm * leaf_mask

    # 3. Cutoff Thresholding: Potong aktivasi residual non-lesi
    hm_thresh = np.maximum(hm - cutoff, 0.0)
    if hm_thresh.max() > 0:
        hm_thresh = hm_thresh / hm_thresh.max()

    # 4. Pewarnaan Colormap JET
    hm_uint8 = (hm_thresh * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)

    # 5. Dynamic Alpha Blending:
    # Area latar belakang & daun sehat memiliki alpha = 0.0 (100% foto asli tanpa noda warna).
    # Area infeksi menyala dengan kontras tajam langsung di atas bercak.
    alpha_mask = np.expand_dims(hm_thresh, axis=-1) * alpha
    blend = (alpha_mask * hm_color + (1.0 - alpha_mask) * orig).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blend)

def make_grayscale_heatmap(heat01: np.ndarray, size_hw: Tuple[int, int]) -> Image.Image:
    w, h = size_hw
    hm = cv2.resize(heat01, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_uint8 = (hm * 255).astype(np.uint8)
    return Image.fromarray(hm_uint8)

def colorize_only(heat01: np.ndarray, size_hw: Tuple[int, int], cutoff: float = 0.15) -> Image.Image:
    w, h = size_hw
    hm = cv2.resize(heat01, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_thresh = np.maximum(hm - cutoff, 0.0)
    if hm_thresh.max() > 0:
        hm_thresh = hm_thresh / hm_thresh.max()
    hm_uint8 = (hm_thresh * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)[:, :, ::-1]
    return Image.fromarray(hm_color)

def get_acts_safe(model: tf.keras.Model, batch: np.ndarray):
    last_conv = _find_last_conv(model)
    if last_conv is None:
        return None, None, "Tidak menemukan conv layer 4D."

    # 1) Direct call pada layer/backbone 4D (Sangat andal untuk Keras 3 Functional sub-model)
    try:
        acts = last_conv(batch, training=False)
        acts = acts.numpy() if hasattr(acts, "numpy") else np.array(acts)
        if acts.ndim == 4:
            return acts, last_conv.name, None
    except Exception:
        pass

    # 2) Fallback: Sub-model graph
    try:
        act_model = tf.keras.Model(model.inputs, last_conv.output)
        acts = act_model(batch, training=False)
        acts = acts.numpy() if hasattr(acts, "numpy") else np.array(acts)
        return acts, last_conv.name, None
    except Exception:
        pass

    # 3) Fallback: Telusuri sub-layer jika last_conv adalah nested model
    if hasattr(last_conv, "layers"):
        for sub in reversed(last_conv.layers):
            try:
                shp = getattr(sub, "output_shape", None)
                if isinstance(shp, tuple) and len(shp) == 4:
                    sub_m = tf.keras.Model(last_conv.inputs, sub.output)
                    acts = sub_m(batch, training=False)
                    acts = acts.numpy() if hasattr(acts, "numpy") else np.array(acts)
                    return acts, f"{last_conv.name}.{sub.name}", None
            except Exception:
                continue

    return None, getattr(last_conv, "name", "unknown"), "Tidak dapat mengekstrak activation map."

# ---------- Cache: Load Model & Encoders Sekali ----------
@st.cache_resource(show_spinner=False)
def load_encoders(plant_pkl_path: Optional[str], disease_pkl_path: Optional[str]):
    def _safe_load(path: Optional[str]):
        if not path or not os.path.exists(path):
            return None, "file tidak ada"
        # 1) Utamakan joblib (standar persistensi sklearn)
        try:
            import joblib
            return joblib.load(path), None
        except Exception as e1:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f), None
            except Exception as e2:
                return None, f"joblib gagal ({e1}), pickle gagal ({e2})"

    p_le, p_err = _safe_load(plant_pkl_path)
    d_le, d_err = _safe_load(disease_pkl_path)

    if p_le is None:
        st.sidebar.warning("Plant encoder tidak dipakai → gunakan DEFAULT_PLANTS. " + (p_err or ""))
    if d_le is None:
        st.sidebar.warning("Disease encoder tidak dipakai → gunakan DEFAULT_DISEASES. " + (d_err or ""))

    return p_le, d_le

@st.cache_resource(show_spinner="Memuat model deep learning...")
def load_fixed_model(path: str):
    if not os.path.exists(path):
        st.error(f"File model tidak ditemukan di: {path}")
        st.stop()
    return load_model(path)

# Load resources
model = load_fixed_model(MODEL_PATH)
plant_le, disease_le = load_encoders(PLANT_PKL_PATH, DISEASE_PKL_PATH)
plant_labels  = list(getattr(plant_le, "classes_", DEFAULT_PLANTS))
disease_labels= list(getattr(disease_le, "classes_", DEFAULT_DISEASES))

if len(model.outputs) != 2:
    st.error(f"Model ini memiliki {len(model.outputs)} output; app ini mengharapkan 2 (plant & disease).")
    st.stop()

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Studio")
    conf_thresh = st.slider("Batas Akurasi Valid (%)", 30, 80, 50, 5, help="Jika akurasi deteksi di bawah batas ini, sistem akan menampilkan status Tidak Terdeteksi / Tidak Dikenali.")
    cam_cutoff = st.slider("Fokus Lesi Heatmap (Cutoff)", 0.0, 0.5, 0.15, 0.05, help="Memotong rembesan aktivasi ke latar belakang/tepi daun agar heatmap terkunci hanya pada fokus infeksi.")
    alpha = st.slider("Intensitas Warna Heatmap", 0.1, 0.9, 0.50, 0.05)
    topk = st.slider("Jumlah Top-K Prediksi", 1, 10, 5)
    img_size_sel = st.select_slider("Ukuran Input Model (px)", options=[128, 160, 192, 224, 256, 299], value=224)
    st.markdown("---")
    st.caption("🤖 **Arsitektur:** NasNetMobile Dual-Head  \n🔬 **Metode XAI:** Eigen-CAM (SVD Activation)  \n🌱 **Cakupan:** 10 Tanaman & 24 Patogen")

# ---------- Mobile Input Selector: Upload Galeri vs Kamera Smartphone ----------
st.markdown("""
<div class="sec-header">
    <div class="sec-bar"></div>
    <div class="sec-title">Pilih Sumber Foto Daun</div>
</div>
""", unsafe_allow_html=True)

input_source = st.radio(
    "Pilih Sumber Foto:",
    ["📁 Upload Berkas / Galeri", "📷 Kamera Smartphone Langsung"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

images_to_process = []

if "Kamera" in input_source:
    st.caption("Arahkan kamera smartphone ke daun yang bergejala sakit, lalu tekan tombol ambil foto.")
    cam_file = st.camera_input("Arahkan kamera ke daun tanaman", label_visibility="collapsed")
    if cam_file is not None:
        images_to_process.append((cam_file, "Foto Kamera Langsung"))
else:
    uploaded_files = st.file_uploader(
        "Pilih atau seret foto daun (Format: JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            images_to_process.append((f, f.name))

# ---------- Tampilan Awal (Belum Ada Gambar Dipilih) ----------
if not images_to_process:
    st.markdown("""
    <div class="sec-header">
        <div class="sec-bar"></div>
        <div class="sec-title">Alur Kerja Diagnosis</div>
    </div>
    <div class="step-list">
        <div class="step-item">
            <div class="step-idx">01</div>
            <div class="step-content">
                <div class="step-head">Ambil atau Unggah Foto Daun</div>
                <div class="step-text">Gunakan kamera ponsel di lapangan atau unggah berkas foto dari galeri perangkat.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-idx">02</div>
            <div class="step-content">
                <div class="step-head">Analisis Dual-Head Neural Network</div>
                <div class="step-text">Model mengidentifikasi varietas komoditas sekaligus mendiagnosa patogen penyakit secara simultan.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-idx">03</div>
            <div class="step-content">
                <div class="step-head">Protokol Penanganan & Verifikasi XAI</div>
                <div class="step-text">Dapatkan rekomendasi tindakan agronomi terpadu beserta visualisasi heatmap Eigen-CAM.</div>
            </div>
        </div>
    </div>

    <div class="sec-header">
        <div class="sec-bar"></div>
        <div class="sec-title">10 Komoditas yang Didukung</div>
    </div>
    <div class="crop-grid">
        <div class="crop-box">🍅 TOMAT</div>
        <div class="crop-box">🍆 TERONG</div>
        <div class="crop-box">🥒 MENTIMUN</div>
        <div class="crop-box">🍈 PARE</div>
        <div class="crop-box">🫘 KACANG PANJANG</div>
        <div class="crop-box">🫛 BUNCIS</div>
        <div class="crop-box">🥦 KEMBANG KOL</div>
        <div class="crop-box">🥢 LOBAK</div>
        <div class="crop-box">🍃 DAUN SIRIH</div>
        <div class="crop-box">🎃 LABU AIR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <details class="guide-accordion">
        <summary class="guide-summary">
            <span>💡 PETUNJUK PEMOTRETAN DAUN YANG OPTIMAL</span>
            <span class="guide-chevron">▾</span>
        </summary>
        <div class="guide-body">
            <div class="guide-row"><span class="guide-bullet">■</span><span><b>Fokus Pada Bercak:</b> Posisikan bercak atau daun sakit di bagian tengah frame kamera.</span></div>
            <div class="guide-row"><span class="guide-bullet">■</span><span><b>Pencahayaan Terang:</b> Ambil foto di bawah cahaya alami yang cukup terang.</span></div>
            <div class="guide-row"><span class="guide-bullet">■</span><span><b>Jarak Ideal:</b> Pertahankan jarak 15–30 cm antara lensa kamera dan helai daun.</span></div>
            <div class="guide-row"><span class="guide-bullet">■</span><span><b>Hindari Blur:</b> Pastikan kamera fokus sebelum mengambil foto.</span></div>
        </div>
    </details>

    <div class="app-footer">
        <b>RimboAI</b> • Platform Diagnosa Kesehatan Tanaman & Explainable AI<br>
        <i>Copyright © 2026 Dede Fauzi, M.Kom, Mahmud Isnan, M.Eng. All rights reserved.</i>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------- Proses Inferensi Gambar Terpilih ----------
for up_file, up_name in images_to_process:
    report_id = f"RBO-{str(uuid.uuid4())[:6].upper()}"
    timestamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M WIB")

    orig = Image.open(up_file).convert("RGB")
    arr = preprocess_pil(orig, (img_size_sel, img_size_sel))
    batch = np.expand_dims(arr, axis=0).astype("float32")

    # Prediksi dua head dengan penghitung waktu inferensi
    t_start = datetime.datetime.now()
    try:
        plant_logits, disease_logits = model(batch, training=False)
    except Exception:
        outs = model(batch, training=False)
        plant_logits, disease_logits = outs[0], outs[1]
    t_end = datetime.datetime.now()
    infer_ms = (t_end - t_start).total_seconds() * 1000.0

    p_probs = ensure_softmax(np.array(plant_logits))
    d_probs = ensure_softmax(np.array(disease_logits))
    p_idx = int(np.argmax(p_probs[0]))
    d_idx = int(np.argmax(d_probs[0]))
    p_lbl = plant_labels[p_idx] if p_idx < len(plant_labels) else f"class_{p_idx}"
    d_lbl = disease_labels[d_idx] if d_idx < len(disease_labels) else f"class_{d_idx}"
    p_conf = p_probs[0][p_idx] * 100
    d_conf = d_probs[0][d_idx] * 100
    is_healthy = "healthy" in d_lbl.lower()

    # Evaluasi status keyakinan (Non-blocking: selalu tampilkan hasil diagnosa)
    is_low_confidence = (p_conf < 40.0) or (d_conf < 35.0)

    # Kartu Hasil Diagnosa (Mobile Layout)
    st.markdown(f"""
    <div class="result-card">
        <div class="result-top">
            <div class="result-title">HASIL DIAGNOSA ({up_name})</div>
            <div class="result-id">ID: {report_id} • {infer_ms:.0f} ms</div>
        </div>
        <div class="result-body">
            <div class="result-metrics">
                <div class="metric-cell">
                    <div class="cell-label">Varietas Tanaman</div>
                    <div class="cell-val">{p_lbl}</div>
                    <span class="cell-pill {'pill-green' if p_conf >= 50 else 'pill-red'}">
                        Akurasi: {p_conf:.1f}%
                    </span>
                </div>
                <div class="metric-cell {'metric-cell-healthy' if is_healthy else 'metric-cell-disease'}">
                    <div class="cell-label">Diagnosa Patogen</div>
                    <div class="cell-val" style="color: {'#15803d' if is_healthy else '#b91c1c'};">{d_lbl}</div>
                    <span class="cell-pill {'pill-green' if is_healthy else 'pill-red'}">
                        {'Kondisi Sehat' if is_healthy else 'Terindikasi Infeksi'} • {d_conf:.1f}%
                    </span>
                </div>
            </div>
            {'<div style="margin-top:10px; padding:8px 10px; background:#fffbeb; border-left:3px solid #f59e0b; font-size:0.78rem; color:#92400e;">⚠️ <b>Catatan:</b> Tingkat keyakinan model di bawah 50%. Hasil diagnosa bersifat indikatif awal; disarankan memotret lebih dekat pada helai daun utama.</div>' if is_low_confidence else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Protokol Resep Pengendalian Fitopatologi
    d_info = get_disease_details(d_lbl)
    rx_items_html = "".join([f'<div class="rx-item"><span class="rx-marker">■</span><span>{step}</span></div>' for step in d_info['penanganan']])

    st.markdown(f"""
    <div class="rx-card">
        <div class="rx-header">🩺 Protokol Penanganan & Rekomendasi Agronomi</div>
        <div class="rx-section">
            <div class="rx-caption">Nama Penyakit & Penyebab</div>
            <div class="rx-val"><b>{d_info['nama_id']}</b> ({d_info['penyebab']})</div>
        </div>
        <div class="rx-section">
            <div class="rx-caption">Karakteristik Gejala Lapangan</div>
            <div class="rx-val">{d_info['gejala']}</div>
        </div>
        <div class="rx-action-box">
            <div class="rx-caption" style="color: #15803d;">Tindakan Pengendalian Terpadu (PHT):</div>
            {rx_items_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Studio Visualisasi Explainable AI (Eigen-CAM)
    st.markdown("""
    <div class="sec-header">
        <div class="sec-bar"></div>
        <div class="sec-title">Studio Transparansi XAI (Eigen-CAM)</div>
    </div>
    <div class="xai-scale">
        <span>🌿 Jaringan Alami (Transparan)</span>
        <div class="scale-bar"></div>
        <span>🔥 Titik Fokus Infeksi</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Visualisasi dilengkapi Dynamic Thresholding: aktivasi di bawah batas ambang dibuat transparan agar tidak meluber ke latar belakang, mengunci fokus langsung pada bercak penyakit.")

    acts, last_name, err = get_acts_safe(model, batch)

    if acts is not None:
        heat = eigen_cam_from_acts(acts)
        heat_gray = make_grayscale_heatmap(heat, orig.size)
        overlay_img = overlay_heatmap_on_img(orig, heat, alpha=alpha, cutoff=cam_cutoff)

        st.markdown("""
        <div class="sec-header">
            <div class="sec-bar"></div>
            <div class="sec-title">EIGEN-CAM (Ori • Heat • Overlay)</div>
        </div>
        """, unsafe_allow_html=True)

        c_ori, c_heat, c_over = st.columns(3)
        with c_ori:
            st.image(orig, caption="Ori (Foto Asli)", use_container_width=True)
        with c_heat:
            st.image(heat_gray, caption="Heat (Aktivasi)", use_container_width=True)
        with c_over:
            st.image(overlay_img, caption="Overlay (Visual XAI)", use_container_width=True)
    else:
        st.warning(f"EigenCAM belum dapat dirender: {err}")
        st.image(orig, caption="Foto Daun Asli", use_container_width=True)

    # Matriks Probabilitas Top-K
    with st.expander("📊 Matriks Probabilitas Lengkap (Top-K)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Komoditas Tanaman**")
            for lbl, p in topk_list(p_probs[0], plant_labels, k=topk):
                st.progress(p, text=f"{lbl}: {p*100:.1f}%")
        with c2:
            st.markdown("**Dugaan Penyakit**")
            for lbl, p in topk_list(d_probs[0], disease_labels, k=topk):
                st.progress(p, text=f"{lbl}: {p*100:.1f}%")

    st.markdown("---")

st.markdown("""
<div class="app-footer">
    <b>RimboAI</b> • Platform Diagnosa Kesehatan Tanaman & Explainable AI<br>
    <i>Copyright © 2026 Dede Fauzi, M.Kom, Mahmud Isnan, M.Eng. All rights reserved.</i>
</div>
""", unsafe_allow_html=True)
