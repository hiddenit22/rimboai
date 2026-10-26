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

# ========= Lokasi file model & label encoders (Relative Path) =========
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "Model", "NasnetMobile+Augment.keras")
PLANT_PKL_PATH   = os.path.join(BASE_DIR, "Encoder", "plant_encoder.pkl")
DISEASE_PKL_PATH = os.path.join(BASE_DIR, "Encoder", "disease_encoder.pkl")
# =====================================================================

st.set_page_config(page_title="Plant & Disease — EigenCAM", layout="centered")
st.title("🌿 Plant & Disease (NasNet) — EigenCAM Prototype")

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
    try:
        act_model = tf.keras.Model(model.inputs, last_conv.output)
        acts = act_model(batch, training=False)
        acts = acts.numpy() if hasattr(acts, "numpy") else np.array(acts)
        return acts, last_conv.name, None
    except Exception as e1:
        try:
            acts = act_model.predict(batch, verbose=0)
            return acts, last_conv.name, None
        except Exception as e2:
            try:
                K = tf.keras.backend
                fn = K.function([model.input], [last_conv.output])
                acts = fn([batch])[0]
                return acts, last_conv.name, None
            except Exception as e3:
                return None, last_conv.name, f"{e1} || {e2} || {e3}"

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

# ---------- sidebar input ----------
uploaded_imgs = st.file_uploader("Upload image(s)...", type=["jpg","jpeg","png"], accept_multiple_files=True)
img_size_sel = st.sidebar.select_slider("Ukuran input", options=[128,160,192,224,256,299], value=224)
alpha = st.sidebar.slider("Transparansi overlay", 0.1, 0.9, 0.35, 0.05)
topk = st.sidebar.slider("Top-K", 1, 10, 5)

if not uploaded_imgs:
    st.info("⬅️ Upload satu atau beberapa gambar untuk mulai.")
    st.stop()

# ---------- proses tiap gambar ----------
# siapkan activation model sekali
acts_demo, last_name, _ = get_acts_safe(model, np.zeros((1, img_size_sel, img_size_sel, 3), dtype=np.float32))
# (kalau gagal di sini, nanti tiap gambar akan coba ulang; ini hanya warm-up)

for up in uploaded_imgs:
    orig = Image.open(up).convert("RGB")
    arr = preprocess_pil(orig, (img_size_sel, img_size_sel))
    batch = np.expand_dims(arr, axis=0).astype("float32")

    # prediksi dua head
    try:
        plant_logits, disease_logits = model(batch, training=False)
    except Exception:
        outs = model(batch, training=False)
        plant_logits, disease_logits = outs[0], outs[1]

    p_probs = ensure_softmax(np.array(plant_logits))
    d_probs = ensure_softmax(np.array(disease_logits))
    p_idx = int(np.argmax(p_probs[0])); d_idx = int(np.argmax(d_probs[0]))
    p_lbl = plant_labels[p_idx] if p_idx < len(plant_labels) else f"class_{p_idx}"
    d_lbl = disease_labels[d_idx] if d_idx < len(disease_labels) else f"class_{d_idx}"

    st.markdown(f"### 📷 {up.name}")
    st.caption(f"Prediksi → **Plant:** {p_lbl} (p={p_probs[0][p_idx]:.3f}) | **Disease:** {d_lbl} (p={d_probs[0][d_idx]:.3f})")

    # tampilkan top-k
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top-K Plant**")
        for lbl, p in topk_list(p_probs[0], plant_labels, k=topk):
            st.progress(p, text=f"{lbl}: {p:.3f}")
    with c2:
        st.markdown("**Top-K Disease**")
        for lbl, p in topk_list(d_probs[0], disease_labels, k=topk):
            st.progress(p, text=f"{lbl}: {p:.3f}")

    # 3 kolom: Ori → Eigen overlay → Heat-only
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.image(orig, caption="Original", use_container_width=True)

    acts, last_name, err = get_acts_safe(model, batch)
    if acts is None:
        with col2: st.warning(f"EigenCAM belum bisa dibuat: {err}")
        with col3: st.info("Heatmap tidak tersedia.")
        continue

    heat = eigen_cam_from_acts(acts)
    overlay_img = overlay_heatmap_on_img(orig, heat, alpha=alpha)
    heat_only   = colorize_only(heat, orig.size)

    with col2:
        st.image(overlay_img, caption=f"Eigen Overlay (last: {last_name})", use_container_width=True)
    with col3:
        st.image(heat_only, caption="Eigen Heatmap", use_container_width=True)

st.markdown("---")
st.caption("EigenCAM bersifat class-agnostic (tidak memilih head). "
           "Jika overlay terlalu kuat/terang, turunkan slider transparansi.")
