from pathlib import Path
import os

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# ============================================================
# Konfigurasi aplikasi
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Tekstur Tanah",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


def cari_checkpoint():
    """Cari checkpoint dari environment variable, folder aplikasi, atau folder output notebook."""
    kandidat = []

    # Bisa diatur dari terminal: set MODEL_PATH=C:\\path\\model_terbaik.pt
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        kandidat.append(Path(env_path).expanduser())

    # Pilihan paling mudah: letakkan model_terbaik.pt di samping app.py
    kandidat.append(BASE_DIR / "model_terbaik.pt")

    # Lokasi output notebook yang terlihat pada contoh pengguna
    kandidat.append(
        Path.home()
        / "Downloads"
        / "output_klasifikasi_tanah"
        / "ringan_undersample"
        / "model_terbaik.pt"
    )

    for path in kandidat:
        if path.is_file():
            return path.resolve()
    return None


def buat_model(nama, n_kelas):
    """Bangun arsitektur yang sama dengan checkpoint; bobot dimuat dari checkpoint."""
    if nama == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_kelas)
    elif nama == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_kelas)
    elif nama == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_kelas)
    elif nama == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_kelas)
    else:
        raise ValueError(f"Arsitektur model belum didukung: {nama}")
    return model


@st.cache_resource(show_spinner="Memuat model klasifikasi tanah...")
def muat_model(path_checkpoint, waktu_modifikasi):
    """Muat model sekali dan simpan di cache Streamlit."""
    del waktu_modifikasi  # dipakai sebagai kunci cache ketika file model berubah

    # Checkpoint ini berasal dari torch.save() pada notebook penelitian sendiri.
    try:
        checkpoint = torch.load(path_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        # Kompatibilitas dengan versi PyTorch lama yang belum memiliki weights_only.
        checkpoint = torch.load(path_checkpoint, map_location="cpu")

    kelas = checkpoint.get("daftar_kelas")
    nama_model = checkpoint.get("model_name", "resnet18")
    ukuran = int(checkpoint.get("img_size", 128))
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])

    if not kelas:
        raise ValueError("Daftar kelas tidak ditemukan di dalam checkpoint.")
    if "model_state" not in checkpoint:
        raise ValueError("Bobot 'model_state' tidak ditemukan di dalam checkpoint.")

    model = buat_model(nama_model, len(kelas))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Harus sama dengan transform_eval pada notebook pelatihan.
    preprocessing = transforms.Compose([
        transforms.Resize((ukuran, ukuran)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    info = {
        "model_name": nama_model,
        "img_size": ukuran,
        "epoch": checkpoint.get("epoch"),
        "val_f1": checkpoint.get("val_f1"),
    }
    return model, kelas, preprocessing, info


@torch.inference_mode()
def prediksi_gambar(model, preprocessing, gambar, kelas):
    """Prediksi satu patch gambar dan kembalikan probabilitas softmax per kelas."""
    gambar_rgb = gambar.convert("RGB")
    tensor_gambar = preprocessing(gambar_rgb).unsqueeze(0)
    logits = model(tensor_gambar)
    probabilitas = torch.softmax(logits.float(), dim=1)[0]
    urutan = torch.argsort(probabilitas, descending=True)

    hasil = [
        {"Kelas": kelas[int(i)], "Probabilitas (%)": float(probabilitas[int(i)]) * 100}
        for i in urutan
    ]
    return hasil


# ============================================================
# Antarmuka
# ============================================================
st.markdown(
    """
    <div style="padding: 0.3rem 0 0.8rem 0;">
      <div style="font-size: 0.88rem; color: #6b7280; letter-spacing: 0.08em; text-transform: uppercase;">
        Computer Vision · Transfer Learning
      </div>
      <h1 style="margin: 0.15rem 0 0.35rem 0;">🌱 Klasifikasi Tekstur Tanah</h1>
      <p style="font-size: 1.05rem; color: #64748b; margin-top: 0;">
        Unggah satu patch citra tanah untuk mendapatkan prediksi kelas Clay, Loam, atau Sand.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Tentang aplikasi")
    st.write("Aplikasi inferensi model CNN berbasis Transfer Learning.")
    st.markdown("**Arsitektur:** ResNet18 (checkpoint penelitian)")
    st.markdown("**Kelas:** Clay · Loam · Sand")
    st.markdown("**Input:** satu patch citra tanah")
    st.divider()
    st.caption(
        "Gunakan gambar patch yang menyerupai data pelatihan. Model ini dilatih pada patch tanah, "
        "bukan untuk menilai foto lanskap atau foto lahan yang sangat luas."
    )

checkpoint_path = cari_checkpoint()

if checkpoint_path is None:
    st.error("File model_terbaik.pt belum ditemukan.")
    st.markdown(
        "Letakkan **model_terbaik.pt** di folder yang sama dengan `app.py`, atau atur environment variable `MODEL_PATH`."
    )
    st.code(
        "Folder aplikasi\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── model_terbaik.pt",
        language="text",
    )
    st.info(
        "Aplikasi juga akan mencari checkpoint pada `Downloads/output_klasifikasi_tanah/"
        "ringan_undersample/model_terbaik.pt`."
    )
    st.stop()

try:
    model, daftar_kelas, preprocessing, info_model = muat_model(
        str(checkpoint_path), checkpoint_path.stat().st_mtime
    )
except Exception as e:
    st.error("Model gagal dimuat. Pastikan checkpoint berasal dari notebook klasifikasi tanah yang sama.")
    st.exception(e)
    st.stop()

# Info model
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Arsitektur", info_model["model_name"].upper())
with m2:
    st.metric("Ukuran input", f'{info_model["img_size"]} × {info_model["img_size"]} px')
with m3:
    val_f1 = info_model.get("val_f1")
    st.metric("F1-macro validasi saat tersimpan", f"{float(val_f1):.4f}" if val_f1 is not None else "Tidak tersedia")

st.divider()
st.subheader("1. Unggah citra tanah")
uploaded_file = st.file_uploader(
    "Pilih satu gambar patch tanah",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
    accept_multiple_files=False,
    help="Format gambar umum didukung. Untuk hasil yang lebih relevan, gunakan patch tanah serupa dengan dataset pelatihan.",
)

if uploaded_file is None:
    st.info("Unggah gambar untuk memulai klasifikasi.")
else:
    try:
        gambar = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("File tidak dapat dibaca sebagai gambar. Coba unggah gambar lain.")
        st.stop()

    st.subheader("2. Hasil prediksi")
    kolom_gambar, kolom_hasil = st.columns([1, 1.15], gap="large")

    with kolom_gambar:
        st.image(gambar, caption=f"Gambar input: {uploaded_file.name}", use_container_width=True)
        st.caption(f"Dimensi gambar asli: {gambar.width} × {gambar.height} piksel")

    with kolom_hasil:
        with st.spinner("Menganalisis tekstur gambar..."):
            hasil = prediksi_gambar(model, preprocessing, gambar, daftar_kelas)

        prediksi_utama = hasil[0]
        st.markdown("**Prediksi utama**")
        st.markdown(f"## {prediksi_utama['Kelas']}")
        st.metric("Skor softmax", f"{prediksi_utama['Probabilitas (%)']:.2f}%")
        st.caption(
            "Skor softmax menunjukkan keyakinan relatif model, bukan jaminan bahwa prediksi benar."
        )

        tabel_hasil = pd.DataFrame(hasil)
        tabel_hasil["Probabilitas (%)"] = tabel_hasil["Probabilitas (%)"].round(2)
        st.markdown("**Probabilitas untuk setiap kelas**")
        st.bar_chart(tabel_hasil.set_index("Kelas"), y="Probabilitas (%)")
        st.dataframe(tabel_hasil, hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Catatan: hasil ini adalah klasifikasi visual berdasarkan pola yang dipelajari dari dataset. "
        "Prediksi tidak menggantikan pengujian laboratorium atau validasi lapangan."
    )

st.sidebar.divider()
st.sidebar.caption(f"Checkpoint: {checkpoint_path.name}")
