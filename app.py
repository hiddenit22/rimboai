# app.py — NasNet (Plant & Disease) + EigenCAM, fixed model path
# ---------------------------------------------------------------
import os, io, pickle
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
    page_title="Rimbo AI — Plant & Disease Diagnosis",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS agar tampilan smartphone responsif, rapi, dan modern
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        max-width: 720px;
    }
    .main-header {
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .main-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #166534;
        margin-bottom: 4px;
    }
    .main-subtitle {
        font-size: 0.9rem;
        color: #4b5563;
        margin-bottom: 8px;
    }
    .diagnosis-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 16px;
        margin: 14px 0;
        box-shadow: 0 4px 12px rgba(22, 101, 52, 0.08);
    }
    .diagnosis-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dashed #bbf7d0;
    }
    .diagnosis-item:last-child {
        border-bottom: none;
    }
    .diag-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1f2937;
    }
    .diag-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #15803d;
    }
    .diag-badge {
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #dcfce7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 12px;
        margin-left: 6px;
    }
    div[data-testid="stCameraInput"] {
        border-radius: 12px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    button {
        border-radius: 10px !important;
    }
    .treatment-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-left: 5px solid #16a34a;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 14px 0 16px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .treatment-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #15803d;
        margin-bottom: 6px;
    }
    .treatment-section {
        margin-top: 6px;
        font-size: 0.92rem;
        color: #374151;
        line-height: 1.45;
    }
    .treatment-label {
        font-weight: 700;
        color: #111827;
    }
</style>
<div class="main-header">
    <div class="main-title">🌿 Rimbo AI — Plant & Disease</div>
    <div class="main-subtitle">Diagnosa Cerdas Tanaman & Penyakit Daun + Visualisasi XAI</div>
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
    st.header("⚙️ Pengaturan")
    alpha = st.slider("Transparansi Overlay Heatmap", 0.1, 0.9, 0.35, 0.05)
    topk = st.slider("Jumlah Top-K Prediksi", 1, 10, 5)
    img_size_sel = st.select_slider("Ukuran Input Model", options=[128, 160, 192, 224, 256, 299], value=224)
    st.markdown("---")
    st.caption("🤖 **Model:** NasNetMobile Dual-Head  \n🔬 **XAI:** Eigen-CAM (Class-agnostic)")

# ---------- pilihan input: Kamera Smartphone vs Galeri ----------
input_source = st.radio(
    "Pilih Sumber Foto:",
    ["📷 Kamera Smartphone Langsung", "📁 Galeri / Upload Berkas"],
    horizontal=True,
    label_visibility="collapsed"
)

images_to_process = []

if "Kamera" in input_source:
    st.caption("👉 Arahkan kamera smartphone Anda ke daun tanaman, lalu tekan tombol foto.")
    cam_file = st.camera_input("Arahkan kamera ke daun tanaman", label_visibility="collapsed")
    if cam_file is not None:
        images_to_process.append((cam_file, "Foto Kamera"))
else:
    uploaded_files = st.file_uploader(
        "Pilih foto daun dari galeri:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            images_to_process.append((f, f.name))

if not images_to_process:
    st.info("💡 Silakan ambil foto langsung dengan kamera atau pilih dari galeri untuk memulai diagnosa.")
    st.stop()

# ---------- proses tiap gambar ----------
for up_file, up_name in images_to_process:
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

    # Kartu Hasil Diagnosa Utama (Elegan & Mobile-friendly)
    st.markdown(f"""
    <div class="diagnosis-card">
        <div style="font-weight: 700; color: #166534; margin-bottom: 6px;">📋 Hasil Diagnosa ({up_name})</div>
        <div class="diagnosis-item">
            <span class="diag-label">🌿 Jenis Tanaman</span>
            <div><span class="diag-val">{p_lbl}</span><span class="diag-badge">{p_conf:.1f}%</span></div>
        </div>
        <div class="diagnosis-item">
            <span class="diag-label">🦠 Dugaan Penyakit</span>
            <div><span class="diag-val" style="color: {'#15803d' if 'healthy' in d_lbl.lower() else '#b91c1c'};">{d_lbl}</span><span class="diag-badge">{d_conf:.1f}%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Narasi Gejala & Rekomendasi Penanganan Agronomi
    d_info = get_disease_details(d_lbl)
    st.markdown(f"""
    <div class="treatment-card">
        <div class="treatment-header">🩺 Informasi & Rekomendasi: {d_info['nama_id']}</div>
        <div class="treatment-section">
            <span class="treatment-label">🔬 Patogen Penyebab:</span> {d_info['penyebab']}
        </div>
        <div class="treatment-section">
            <span class="treatment-label">🔍 Gejala Khas:</span> {d_info['gejala']}
        </div>
        <div class="treatment-section" style="border-top: 1px dashed #cbd5e1; padding-top: 8px; margin-top: 8px;">
            <span class="treatment-label" style="color: #15803d;">💊 Langkah Pengendalian & Penanganan:</span>
            <ul style="margin: 6px 0 0 18px; padding: 0;">
                {''.join([f'<li style="margin-bottom: 4px;">{step}</li>' for step in d_info['penanganan']])}
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visualisasi XAI & Foto (Menggunakan Tab Responsif untuk Smartphone)
    acts, last_name, err = get_acts_safe(model, batch)

    if acts is not None:
        heat = eigen_cam_from_acts(acts)
        overlay_img = overlay_heatmap_on_img(orig, heat, alpha=alpha)
        heat_only   = colorize_only(heat, orig.size)

        tab_overlay, tab_heat, tab_orig = st.tabs(["✨ Visualisasi Overlay XAI", "🔥 Heatmap Saja", "📷 Foto Asli"])
        with tab_overlay:
            st.image(overlay_img, caption="Area merah/kuning adalah fokus model dalam mendeteksi gejala.", use_container_width=True)
        with tab_heat:
            st.image(heat_only, caption=f"Peta Aktivasi Konvolusi ({last_name})", use_container_width=True)
        with tab_orig:
            st.image(orig, caption="Foto Daun Asli", use_container_width=True)
    else:
        st.warning(f"EigenCAM belum dapat dirender: {err}")
        st.image(orig, caption="Foto Daun Asli", use_container_width=True)

    # Detail Probabilitas Top-K
    with st.expander("📊 Lihat Detail Probabilitas Lainnya (Top-K)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top-K Tanaman**")
            for lbl, p in topk_list(p_probs[0], plant_labels, k=topk):
                st.progress(p, text=f"{lbl}: {p*100:.1f}%")
        with c2:
            st.markdown("**Top-K Penyakit**")
            for lbl, p in topk_list(d_probs[0], disease_labels, k=topk):
                st.progress(p, text=f"{lbl}: {p*100:.1f}%")

    st.markdown("---")

st.caption("ℹ️ Tips: Buka menu samping (ikon panah kiri atas) untuk mengatur transparansi overlay atau Top-K.")
