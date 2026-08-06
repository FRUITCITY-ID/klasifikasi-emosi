# =============================================================================
# SIPEMO — image backend untuk Railway
#
# Yang dilayani image ini hanyalah API (/api/*). Antarmukanya sendiri di-host
# terpisah di GitHub Pages, jadi frontend/ ikut disalin hanya sebagai cadangan:
# membuka URL Railway langsung tetap memberi UI yang berfungsi kalau Pages
# bermasalah saat sidang.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    SIPEMO_FORCE_CPU=1

WORKDIR /app

# torch dipasang lebih dulu dari indeks CPU milik PyTorch. Wheel default di
# PyPI membawa runtime CUDA sekitar 2,5 GiB — tidak ada gunanya di Railway yang
# tanpa GPU, dan cukup besar untuk membuat build gagal kehabisan ruang.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.5.1"

# requirements.txt menuliskan torch>=2.1; pip melihat 2.5.1+cpu sudah terpasang
# dan melewatinya, jadi baris di atas tidak tertimpa wheel CUDA.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Bobot dasar IndoBERT-base dimasukkan ke dalam image. Kalau tidak, kontainer
# yang baru bangun harus menarik ~500 MiB dari Hugging Face sebelum bisa
# menjawab permintaan pertama — dan kalau hf.co sedang bermasalah saat sidang,
# demonya ikut mati.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1'); \
AutoModelForSequenceClassification.from_pretrained( \
    'indobenchmark/indobert-base-p1', \
    num_labels=4, \
    problem_type='multi_label_classification')"

# Checkpoint hasil training (~475 MiB) diambil dari aset GitHub Release. Nilai
# default ada di backend/fetch_checkpoint.py; ARG di bawah hanya untuk
# menimpanya tanpa mengubah kode.
ARG SIPEMO_CHECKPOINT_URL="https://github.com/FRUITCITY-ID/klasifikasi-emosi/releases/download/v1.0-model/best_indobert_base.pt"
ARG SIPEMO_CHECKPOINT_SHA256="48168bf6e294768c1b567c262d82f473c51ef2becae51a29e08c7cc6499d8e5c"
RUN SIPEMO_CHECKPOINT_URL="${SIPEMO_CHECKPOINT_URL}" \
    SIPEMO_CHECKPOINT_SHA256="${SIPEMO_CHECKPOINT_SHA256}" \
    python /app/backend/fetch_checkpoint.py

# Gate build: kalau langkah di atas gagal mengunduh, image ini tidak boleh
# terbit. Server yang naik tanpa bobot hanya akan melaporkan 'no_checkpoint'
# di /api/health — lebih baik ketahuan saat build daripada saat sidang.
RUN test -s /app/models/best_indobert_base.pt

EXPOSE 8000

# Railway menyuntikkan $PORT. Satu worker saja: tiap worker memuat salinan
# model sendiri (~600 MiB), dan beban demo sidang jauh dari butuh dua.
CMD ["sh", "-c", "uvicorn app:app --app-dir /app/backend --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
