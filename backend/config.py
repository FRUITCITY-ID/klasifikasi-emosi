"""
SIPEMO — konfigurasi sistem.

Semua nilai di file ini disalin apa adanya dari notebook penelitian
`multilabel_bert_comparison.ipynb` agar backend berperilaku identik dengan
eksperimen. Jangan ubah nilai di sini tanpa menjalankan ulang notebook.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Label (notebook cell 1: LABEL_COLS)
# --------------------------------------------------------------------------
LABEL_COLS: list[str] = ["senang", "sedih", "marah", "takut"]

# Metadata tampilan — tidak memengaruhi inferensi, hanya untuk UI.
LABEL_META: dict[str, dict[str, str]] = {
    "senang": {"glyph": "😊", "desc": "Gembira, syukur, puas, lega"},
    "sedih": {"glyph": "😢", "desc": "Kehilangan, kecewa, hampa, pilu"},
    "marah": {"glyph": "😠", "desc": "Kesal, benci, geram, muak"},
    "takut": {"glyph": "😨", "desc": "Cemas, khawatir, panik, gelisah"},
}

# --------------------------------------------------------------------------
# Hyperparameter (notebook cell 1)
# --------------------------------------------------------------------------
SEED = 42
MAX_LEN = 96
BATCH = 16
EPOCHS = 3
LR = 2e-5

# --------------------------------------------------------------------------
# Model yang dilayani
# --------------------------------------------------------------------------
# Notebook cell 6 melatih 6 model. Cell 14 memilih model terbaik menurut
# F1-macro pada test set: "IndoBERT-base (baseline)" (F1_macro = 0.9945).
SERVED_MODEL_NAME = "IndoBERT-base (baseline)"
SERVED_MODEL_ID = "indobenchmark/indobert-base-p1"

# Lebar matriks embedding IndoBERT-base-p1 saat dilatih.
#
# Perhatikan: config.vocab_size model ini 50.000, sedangkan tokenizer-nya hanya
# punya 30.521 token — ketidakcocokan bawaan rilis IndoBERT. Karena cell 9
# melatih tanpa resize_token_embeddings(), checkpoint yang benar berukuran
# 50.000, bukan 30.521.
#
# Nilai ini hanya informatif; gate verifikasi di model.py membandingkan
# checkpoint dengan model yang benar-benar dimuat saat runtime.
EXPECTED_VOCAB_SIZE = 50000

# Threshold hasil tuning pada validation set untuk model terbaik
# (notebook cell 11: threshold = 0.30). Nilai dari checkpoint akan menimpa ini
# bila key 'thr' tersedia DAN checkpoint lolos verifikasi.
DEFAULT_THRESHOLD = 0.30

# --------------------------------------------------------------------------
# Path
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_DIR = PROJECT_DIR / "models"


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    """Resolve path konfigurasi; path relatif selalu berakar di folder proyek."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


# Default portabel. Override checkpoint yang disimpan di luar proyek:
#   set SIPEMO_CHECKPOINT=D:\path\ke\best_indobert_base.pt
CHECKPOINT_PATH = resolve_project_path(
    os.environ.get("SIPEMO_CHECKPOINT", "models/best_indobert_base.pt")
)

# Set SIPEMO_ALLOW_MISMATCH=1 untuk memaksa memuat checkpoint yang arsitekturnya
# TIDAK cocok. Tidak disarankan: metrik yang ditampilkan UI adalah metrik
# IndoBERT-base, jadi memuat bobot model lain akan membuat angka itu bohong.
ALLOW_MISMATCH = os.environ.get("SIPEMO_ALLOW_MISMATCH", "0") == "1"

# Paksa CPU walaupun CUDA tersedia.
FORCE_CPU = os.environ.get("SIPEMO_FORCE_CPU", "0") == "1"

HOST = os.environ.get("SIPEMO_HOST", "127.0.0.1")
# Railway (dan sebagian besar PaaS lain) menyuntikkan $PORT. SIPEMO_PORT tetap
# didahulukan supaya setelan lokal tidak berubah artinya saat proyek ini
# kebetulan dijalankan di lingkungan yang punya $PORT.
PORT = int(os.environ.get("SIPEMO_PORT") or os.environ.get("PORT") or "8000")

# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------
# Saat dijalankan lokal, frontend disajikan oleh server yang sama sehingga
# tidak ada permintaan lintas-origin sama sekali. Yang membutuhkan daftar ini
# adalah deployment: antarmuka ada di GitHub Pages, API di Railway, jadi dua
# origin yang berbeda.
#
# Daftarnya sengaja eksplisit, bukan "*". allow_origins=["*"] akan membuat API
# ini bisa dipanggil dari halaman mana pun di internet, dan meski endpoint-nya
# tidak menyimpan apa pun, tiap panggilan tetap menjalankan inferensi BERT di
# atas kuota Railway milik pemilik proyek.
DEFAULT_CORS_ORIGINS: list[str] = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "https://fruitcity-id.github.io",
]

# Tambah origin lain lewat SIPEMO_CORS_ORIGINS, dipisah koma.
CORS_ORIGINS: list[str] = DEFAULT_CORS_ORIGINS + [
    o.strip()
    for o in os.environ.get("SIPEMO_CORS_ORIGINS", "").split(",")
    if o.strip()
]
