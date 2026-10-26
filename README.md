---
title: Plant & Disease EigenCAM
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: mit
short_description: Dual-head NasNetMobile for plant & disease detection with EigenCAM
---

# 🌿 Plant & Disease Detection with EigenCAM (NasNetMobile)

Prototipe deteksi jenis tanaman (*plant*) dan penyakit (*disease*) berbasis Deep Learning menggunakan arsitektur **NasNetMobile (Dual-Head)** dan divisualisasikan dengan Explainable AI (XAI) **Eigen-CAM**.

## 📌 Fitur Utama
1. **Dual-Head Output:** Memprediksi jenis tanaman dan penyakit tanaman secara simultan.
2. **Top-K Prediction:** Menampilkan probabilitas prediksi teratas dengan visual progress bar.
3. **Eigen-CAM Visualization:** Menghasilkan heatmap atensi visual untuk melihat area daun yang menjadi fokus model (Class-agnostic).
4. **Interactive Controls:** Pengaturan resolusi input gambar, transparansi heatmap overlay, dan Top-K slider.

## 🛠️ Model & Arsitektur
- **Backbone:** NasNetMobile + Data Augmentation
- **Heads:** 
  - Head 1: Plant Classification (10 Kelas)
  - Head 2: Disease Classification (24 Kelas)
- **XAI:** Eigen-CAM (Principal Component Analysis via SVD pada konvolusi layer terakhir)
