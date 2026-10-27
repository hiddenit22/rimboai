# app.py — NasNet (Plant & Disease) + EigenCAM, fixed model path
# ---------------------------------------------------------------
import os, io, pickle, uuid, datetime
from typing import Optional, Tuple, List

import streamlit as st
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
    page_title="RimboAI — Smart Plant Health Diagnostics",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Design System: Google Font, Glassmorphism, Animated Badges & Clean Cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 780px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 45%, #047857 100%);
        border-radius: 20px;
        padding: 26px 20px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 12px 28px -6px rgba(6, 78, 59, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -40px;
        right: -40px;
        width: 120px;
        height: 120px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.25) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 5px 14px;
        border-radius: 30px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #a7f3d0;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        line-height: 1.15;
    }
    .hero-title span {
        color: #34d399;
    }
    .hero-desc {
        font-size: 0.92rem;
        color: #e6fcf5;
        max-width: 580px;
        margin: 0 auto 16px auto;
        line-height: 1.5;
        font-weight: 400;
    }
    .stats-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .stat-pill {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        color: #ffffff;
    }
    .stat-pill b {
        color: #6ee7b7;
    }

    /* Section & Cards */
    .section-head {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin: 22px 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
        margin-bottom: 10px;
    }
    .step-card:hover {
        border-color: #10b981;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.1);
        transform: translateY(-2px);
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background: #ecfdf5;
        color: #059669;
        font-weight: 800;
        font-size: 0.8rem;
        border-radius: 50%;
        margin-bottom: 8px;
    }
    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .step-desc {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.45;
    }

    /* Crop Tags */
    .crop-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 10px 0 16px 0;
    }
    .crop-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 5px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
        transition: all 0.15s ease;
    }
    .crop-pill:hover {
        background: #ecfdf5;
        border-color: #6ee7b7;
        color: #047857;
    }

    /* Clinical Diagnosis Card */
    .clinic-card {
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid #cbd5e1;
        overflow: hidden;
        margin: 16px 0;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.08);
    }
    .clinic-header {
        background: linear-gradient(135deg, #047857 0%, #059669 100%);
        color: #ffffff;
        padding: 14px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .clinic-title {
        font-weight: 700;
        font-size: 1.05rem;
    }
    .clinic-id {
        font-size: 0.75rem;
        background: rgba(255, 255, 255, 0.2);
        padding: 3px 10px;
        border-radius: 20px;
        font-family: monospace;
    }
    .clinic-body {
        padding: 16px 18px;
    }
    .clinic-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }
    @media (max-width: 600px) {
        .clinic-grid {
            grid-template-columns: 1fr;
        }
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .metric-box-disease {
        background: #fff5f5;
        border: 1px solid #fecaca;
    }
    .metric-box-healthy {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 2px;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        margin-top: 4px;
    }

    /* Treatment Box */
    .rx-box {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-left: 5px solid #059669;
        border-radius: 14px;
        padding: 16px;
        margin: 14px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .rx-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #065f46;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .rx-row {
        margin-bottom: 8px;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #334155;
    }
    .rx-tag {
        font-weight: 700;
        color: #0f172a;
    }

    /* XAI Studio */
    .xai-legend {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.78rem;
        color: #475569;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .gradient-bar {
        height: 8px;
        flex-grow: 1;
        margin: 0 10px;
        border-radius: 4px;
        background: linear-gradient(90deg, #1e3a8a 0%, #06b6d4 25%, #eab308 60%, #ef4444 100%);
    }

    /* Tab & Inputs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.88rem;
        border-radius: 8px;
    }
    div[data-testid="stCameraInput"] {
        border-radius: 14px;
        overflow: hidden;
        border: 2px dashed #10b981;
    }
    button {
        border-radius: 10px !important;
    }

    /* Footer */
    .pro-footer {
        text-align: center;
        padding: 28px 10px 14px 10px;
        color: #94a3b8;
        font-size: 0.82rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 36px;
        line-height: 1.6;
    }
    .pro-footer b {
        color: #475569;
    }
</style>

<!-- Hero Section -->
<div class="hero-banner">
    <div class="hero-badge">
        <span class="pulse-dot"></span> SMART AGRITECH & PHYTOPATHOLOGY AI
    </div>
    <div class="hero-title">Rimbo<span>AI</span></div>
    <div class="hero-desc">
        Sistem Pakar Diagnosis Kesehatan Tanaman & Deteksi Dini Penyakit Daun berbasis Deep Learning Dual-Head dengan Transparansi Explainable AI (Eigen-CAM).
    </div>
    <div class="stats-row">
        <div class="stat-pill">🌿 <b>10</b> Komoditas</div>
        <div class="stat-pill">🔬 <b>24</b> Jenis Penyakit</div>
        <div class="stat-pill">⚡ <b>0.3s</b> Inferensi CPU</div>
        <div class="stat-pill">👁️ <b>Eigen-CAM</b> XAI Visual</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- util umum ----------
DEFAULT_PLANTS = ["Bean","Betel Leaf","Bitter Gourd","Bottle gourd","Cauliflower",
                  "Cowpea","Cucumber","Eggplant","Radish","Tomato"]
DEFAULT_DISEASES = [
    "Anthracnose","Bacterial blight","Bacterial wilt","Black rot","Downy mildew",
    "Early blight","Fusarium wilt","Leaf blight","Leaf curl","Leaf rot","Leaf spot",
    "Mildew","Mosaic virus","Powdery mildew","Rust","Septoria","Spot","Virus",
    "Yellow leaf","Healthy","Canker","Scab","Stem rot","Tip burn"
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

# ---------- EIGEN-CAM helper (class-agnostic) ----------
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
    heat = (X @ pc1).reshape(H, W)
    heat = np.abs(heat)
    m = heat.max()
    if m > 0: heat = heat / m
    return heat

def overlay_heatmap_on_img(orig_img: Image.Image, heat01: np.ndarray, alpha: float = 0.35) -> Image.Image:
    orig = np.asarray(orig_img.convert("RGB"))
    h, w = orig.shape[:2]
    hm = (heat01 * 255).astype("uint8")
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    blend = (alpha * hm_color + (1 - alpha) * orig).astype("uint8")
    return Image.fromarray(blend)

def colorize_only(heat01: np.ndarray, size_hw: Tuple[int, int]) -> Image.Image:
    w, h = size_hw
    hm = (heat01 * 255).astype("uint8")
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[:, :, ::-1]
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

    # 3) Fallback: telusuri sub-layer jika last_conv adalah nested model
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

# ---------- cache: load model & encoders sekali ----------
@st.cache_resource(show_spinner=False)
def load_encoders(plant_pkl_path: Optional[str], disease_pkl_path: Optional[str]):
    import pickle
    from pickle import UnpicklingError

    def _safe_load(path: Optional[str]):
        if not path or not os.path.exists(path):
            return None, "file tidak ada"
        # 1) coba pickle standar
        try:
            with open(path, "rb") as f:
                return pickle.load(f), None
        except UnpicklingError as e1:
            # 2) fallback: joblib (sering dipakai untuk encoder sklearn)
            try:
                import joblib
                return joblib.load(path), None
            except Exception as e2:
                return None, f"unpickle gagal ({type(e1).__name__}), joblib juga gagal ({type(e2).__name__})"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

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


# load once
model = load_fixed_model(MODEL_PATH)
plant_le, disease_le = load_encoders(PLANT_PKL_PATH, DISEASE_PKL_PATH)
plant_labels  = list(getattr(plant_le, "classes_", DEFAULT_PLANTS))
disease_labels= list(getattr(disease_le, "classes_", DEFAULT_DISEASES))

# validasi output
if len(model.outputs) != 2:
    st.error(f"Model ini memiliki {len(model.outputs)} output; app ini mengharapkan 2 (plant & disease).")
    st.stop()

# ---------- sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Pengaturan Studio")
    alpha = st.slider("Transparansi Overlay Heatmap", 0.1, 0.9, 0.35, 0.05)
    topk = st.slider("Jumlah Top-K Prediksi", 1, 10, 5)
    img_size_sel = st.select_slider("Ukuran Input Model (px)", options=[128, 160, 192, 224, 256, 299], value=224)
    st.markdown("---")
    st.caption("🤖 **Arsitektur:** NasNetMobile Dual-Head  \n🔬 **Metode XAI:** Eigen-CAM (SVD Activation)  \n🌱 **Corpus:** 10 Tanaman & 24 Patogen")

# ---------- pilihan input: Upload Berkas (Default) vs Kamera Smartphone ----------
st.markdown('<div class="section-head">📸 Pilih Metode Input Foto Daun</div>', unsafe_allow_html=True)
input_source = st.radio(
    "Pilih Sumber Foto:",
    ["📁 Upload Berkas / Galeri", "📷 Kamera Smartphone Langsung"],
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)

images_to_process = []

if "Kamera" in input_source:
    st.caption("👉 Arahkan kamera smartphone ke daun tanaman yang bergejala sakit, lalu tekan tombol ambil foto.")
    cam_file = st.camera_input("Arahkan kamera ke daun tanaman", label_visibility="collapsed")
    if cam_file is not None:
        images_to_process.append((cam_file, "Foto Kamera"))
else:
    uploaded_files = st.file_uploader(
        "Pilih atau seret foto daun (Format: JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            images_to_process.append((f, f.name))

# ---------- Landing Page Showcase jika belum ada gambar ----------
if not images_to_process:
    st.markdown('<div class="section-head">🚀 Alur Kerja Diagnosis Cerdas</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">1</div>
            <div class="step-title">Ambil Foto Daun</div>
            <div class="step-desc">Unggah foto dari galeri atau potret langsung daun di kebun via kamera ponsel.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">2</div>
            <div class="step-title">Analisis Dual-Head AI</div>
            <div class="step-desc">Neural network NasNetMobile mendeteksi spesies tanaman & memprediksi jenis patogen.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <div class="step-num">3</div>
            <div class="step-title">Resep & Peta XAI</div>
            <div class="step-desc">Dapatkan protokol penanganan terpadu serta visualisasi heatmap Eigen-CAM transparan.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">🌿 10 Komoditas Tanaman yang Didukung</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="crop-grid">
        <span class="crop-pill">🍅 Tomat (Tomato)</span>
        <span class="crop-pill">🍆 Terong (Eggplant)</span>
        <span class="crop-pill">🥒 Mentimun (Cucumber)</span>
        <span class="crop-pill">🍈 Pare (Bitter Gourd)</span>
        <span class="crop-pill">🫘 Kacang Panjang (Cowpea)</span>
        <span class="crop-pill">🫛 Buncis (Bean)</span>
        <span class="crop-pill">🥦 Kembang Kol (Cauliflower)</span>
        <span class="crop-pill">🥢 Lobak (Radish)</span>
        <span class="crop-pill">🍃 Daun Sirih (Betel Leaf)</span>
        <span class="crop-pill">🎃 Labu Air (Bottle Gourd)</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 Panduan Pengambilan Foto Terbaik untuk Hasil Akurat", expanded=False):
        st.markdown("""
        * **Fokus pada Gejala Sakit:** Pastikan bercak daun atau tanda penyakit berada di tengah bidang foto.
        * **Pencahayaan Alami Cukup:** Ambil foto dengan cahaya terang alami dan hindari bayangan tangan/badan.
        * **Latar Belakang Bersih:** Hindari foto yang terlalu jauh atau memuat banyak tanaman lain sekaligus.
        * **Jarak Pemotretan:** Jarak ideal kamera dengan permukaan daun adalah sekitar 15 – 30 cm.
        """)

    st.markdown("""
    <div class="pro-footer">
        <b>RimboAI</b> • Platform Diagnosis Kesehatan Tanaman Berbasis Explainable AI (XAI) & Deep Learning<br>
        <i>Prototype Riset Publikasi & Inovasi Pertanian Presisi Digital © 2026</i>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------- Proses Tiap Gambar Terupload / Terfoto ----------
for up_file, up_name in images_to_process:
    report_id = f"RBO-{str(uuid.uuid4())[:6].upper()}"
    timestamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M WIB")

    orig = Image.open(up_file).convert("RGB")
    arr = preprocess_pil(orig, (img_size_sel, img_size_sel))
    batch = np.expand_dims(arr, axis=0).astype("float32")

    # Prediksi dua head
    try:
        plant_logits, disease_logits = model(batch, training=False)
    except Exception:
        outs = model(batch, training=False)
        plant_logits, disease_logits = outs[0], outs[1]

    p_probs = ensure_softmax(np.array(plant_logits))
    d_probs = ensure_softmax(np.array(disease_logits))
    p_idx = int(np.argmax(p_probs[0]))
    d_idx = int(np.argmax(d_probs[0]))
    p_lbl = plant_labels[p_idx] if p_idx < len(plant_labels) else f"class_{p_idx}"
    d_lbl = disease_labels[d_idx] if d_idx < len(disease_labels) else f"class_{d_idx}"
    p_conf = p_probs[0][p_idx] * 100
    d_conf = d_probs[0][d_idx] * 100
    is_healthy = "healthy" in d_lbl.lower()

    # Kartu Hasil Diagnosa Klinis
    st.markdown(f"""
    <div class="clinic-card">
        <div class="clinic-header">
            <div class="clinic-title">📋 Laporan Diagnosa Fitopatologi ({up_name})</div>
            <div class="clinic-id">ID: {report_id} • {timestamp}</div>
        </div>
        <div class="clinic-body">
            <div class="clinic-grid">
                <div class="metric-box">
                    <div class="metric-label">🌿 Komoditas Tanaman</div>
                    <div class="metric-val">{p_lbl}</div>
                    <span class="metric-badge" style="background:#dcfce7; color:#15803d; border:1px solid #86efac;">
                        Akurasi Model: {p_conf:.1f}%
                    </span>
                </div>
                <div class="metric-box {'metric-box-healthy' if is_healthy else 'metric-box-disease'}">
                    <div class="metric-label">🦠 Hasil Diagnosa Patogen</div>
                    <div class="metric-val" style="color: {'#15803d' if is_healthy else '#b91c1c'};">{d_lbl}</div>
                    <span class="metric-badge" style="background:{'#dcfce7' if is_healthy else '#fee2e2'}; color:{'#15803d' if is_healthy else '#b91c1c'}; border:1px solid {'#86efac' if is_healthy else '#fca5a5'};">
                        {'Kondisi Sehat' if is_healthy else 'Terindikasi Infeksi'} • Kepastian: {d_conf:.1f}%
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Protokol Resep Pengendalian Fitopatologi
    d_info = get_disease_details(d_lbl)
    st.markdown(f"""
    <div class="rx-box">
        <div class="rx-header">
            🩺 <b>Protokol Pengendalian & Panduan Agronomi Terpadu</b>
        </div>
        <div class="rx-row">
            <span class="rx-tag">🔬 Nama Penyakit & Etiologi:</span> <b>{d_info['nama_id']}</b> ({d_info['penyebab']})
        </div>
        <div class="rx-row">
            <span class="rx-tag">🔍 Gejala Klinis Lapangan:</span> {d_info['gejala']}
        </div>
        <div class="rx-row" style="border-top: 1px dashed #cbd5e1; padding-top: 10px; margin-top: 10px;">
            <span class="rx-tag" style="color: #047857;">💊 Langkah Pengendalian Terpadu (PHT):</span>
            <ul style="margin: 6px 0 0 18px; padding: 0;">
                {''.join([f'<li style="margin-bottom: 5px;">{step}</li>' for step in d_info['penanganan']])}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Studio Visualisasi Explainable AI (Eigen-CAM)
    st.markdown('<div class="section-head">👁️ Studio Transparansi Visual (Eigen-CAM XAI)</div>', unsafe_allow_html=True)
    st.caption("Visualisasi atensi konvolusi memperlihatkan area daun spesifik yang menjadi dasar utama model dalam mendiagnosa penyakit.")

    st.markdown("""
    <div class="xai-legend">
        <span>🟦 Area Daun Sehat / Latar</span>
        <div class="gradient-bar"></div>
        <span>🟥 Fokus Infeksi Patogen</span>
    </div>
    """, unsafe_allow_html=True)

    acts, last_name, err = get_acts_safe(model, batch)

    if acts is not None:
        heat = eigen_cam_from_acts(acts)
        overlay_img = overlay_heatmap_on_img(orig, heat, alpha=alpha)
        heat_only   = colorize_only(heat, orig.size)

        tab_overlay, tab_heat, tab_orig = st.tabs([
            "✨ Overlay Atensi AI", 
            "🔥 Peta Panas (Heatmap Murni)", 
            "📷 Foto Asli Daun"
        ])
        with tab_overlay:
            st.image(overlay_img, caption=f"Visualisasi Overlay Eigen-CAM (Layer: {last_name})", use_container_width=True)
        with tab_heat:
            st.image(heat_only, caption="Aktivasi Fitur Konvolusi 4D Terakhir", use_container_width=True)
        with tab_orig:
            st.image(orig, caption="Foto Daun yang Dianalisis", use_container_width=True)
    else:
        st.warning(f"EigenCAM belum dapat dirender: {err}")
        st.image(orig, caption="Foto Daun Asli", use_container_width=True)

    # Matriks Probabilitas Lengkap (Top-K)
    with st.expander("📊 Matriks Probabilitas Lengkap (Top-K Candidates)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Probabilitas Varietas Tanaman**")
            for lbl, p in topk_list(p_probs[0], plant_labels, k=topk):
                st.progress(p, text=f"{lbl}: {p*100:.1f}%")
        with c2:
            st.markdown("**Probabilitas Dugaan Penyakit**")
            for lbl, p in topk_list(d_probs[0], disease_labels, k=topk):
                st.progress(p, text=f"{lbl}: {p*100:.1f}%")

    st.markdown("---")

st.markdown("""
<div class="pro-footer">
    <b>RimboAI</b> • Platform Diagnosis Kesehatan Tanaman Berbasis Explainable AI (XAI) & Deep Learning<br>
    <i>Prototype Riset Publikasi & Inovasi Pertanian Presisi Digital © 2026</i>
</div>
""", unsafe_allow_html=True)
