r"""
Verifikasi backend end-to-end (skrip sementara untuk pemeriksaan lokal).

Yang diuji:
  1. Checkpoint berarsitektur lain (vocab mBERT) HARUS ditolak.
  2. Checkpoint IndoBERT-base yang sah HARUS diterima, threshold terbaca.
  3. predict() menghasilkan struktur lengkap dan fusi OR-nya benar.

Bobot yang dipakai di langkah 2-3 adalah bobot acak hasil inisialisasi — jadi
angka probabilitasnya TIDAK bermakna. Yang diverifikasi di sini adalah mekanika
pipeline, bukan kualitas prediksi.

Jalankan dari folder backend/:
    ..\.venv\Scripts\python.exe verify_backend.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

import config
import model as M

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


tmp = Path(tempfile.mkdtemp(prefix="sipemo_verify_"))
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'GAGAL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print("=" * 68)
print("Verifikasi backend SIPEMO")
print("=" * 68)

print("\nMengunduh / memuat", config.SERVED_MODEL_ID, "…")
base = AutoModelForSequenceClassification.from_pretrained(
    config.SERVED_MODEL_ID, num_labels=4, problem_type="multi_label_classification"
)
real_sd = base.state_dict()
real_vocab = int(base.get_input_embeddings().weight.shape[0])
print(f"vocab IndoBERT-base = {real_vocab:,}")

# ---------------------------------------------------------------------------
print("\n[1] Checkpoint arsitektur lain (vocab mBERT 119.547) harus DITOLAK")
# ---------------------------------------------------------------------------
fake = {k: v.clone() for k, v in real_sd.items()}
fake["bert.embeddings.word_embeddings.weight"] = torch.zeros(119547, 768)
bad_path = tmp / "best_bert_mbert.pt"
torch.save({"state_dict": fake, "thr": 0.30, "f1_macro": 0.7816}, bad_path)

config.CHECKPOINT_PATH = bad_path
st = M.load()
check("status == 'mismatch'", st.status == "mismatch", f"dapat '{st.status}'")
check("loaded == False", st.loaded is False)
check("pesan menyebut mBERT", "mBERT" in st.message)
check("pesan menyebut sebab di notebook", "train_one()" in st.message)
check("predict() ditolak", not st.loaded)

# ---------------------------------------------------------------------------
print("\n[2] Checkpoint IndoBERT-base yang sah harus DITERIMA")
# ---------------------------------------------------------------------------
good_path = tmp / "best_indobert_base.pt"
torch.save({"state_dict": real_sd, "thr": 0.30, "f1_macro": 0.9789}, good_path)

config.CHECKPOINT_PATH = good_path
st = M.load()
check("status == 'ready'", st.status == "ready", f"dapat '{st.status}' — {st.message}")
check("loaded == True", st.loaded is True)
check("threshold == 0.30 dari checkpoint", abs(st.threshold - 0.30) < 1e-9,
      f"{st.threshold} · sumber: {st.threshold_source}")
check("tanpa peringatan sidik jari F1", not any("F1-macro" in w for w in st.warnings),
      f"warnings={len(st.warnings)}")

# ---------------------------------------------------------------------------
print("\n[3] Sidik jari F1 yang keliru harus memunculkan PERINGATAN")
# ---------------------------------------------------------------------------
odd_path = tmp / "best_wrong_f1.pt"
torch.save({"state_dict": real_sd, "thr": 0.40, "f1_macro": 0.8658}, odd_path)
config.CHECKPOINT_PATH = odd_path
st = M.load()
check("tetap ready (peringatan, bukan penolakan)", st.status == "ready")
check("peringatan menunjuk IndoBERTweet",
      any("IndoBERTweet" in w for w in st.warnings),
      st.warnings[0][:90] if st.warnings else "tidak ada peringatan")

# ---------------------------------------------------------------------------
print("\n[4] predict() — struktur & logika fusi")
# ---------------------------------------------------------------------------
config.CHECKPOINT_PATH = good_path
M.load()

CASES = [
    "Akhirnya wisuda juga! Bahagia banget 🥳 perjuangan selama ini terbayar",
    "@budi cek https://contoh.co.id ya #SkripsiLife gw udh seneng bgt",
    "Sedih banget kehilangan sahabat 😭 rasanya hampa",
    "Tidak ada emoji di sini",
]

import preprocessing as pp

for text in CASES:
    r = M.predict(text)
    tag = text[:42] + ("…" if len(text) > 42 else "")

    check(f"norm sesuai notebook · {tag}", r["norm"] == pp.normalize(text), r["norm"])
    check(f"4 probabilitas dalam [0,1] · {tag}",
          all(0.0 <= r["probs"][c] <= 1.0 for c in config.LABEL_COLS))
    check(f"bin_bert = sigma >= thr · {tag}",
          all(r["pred_bert"][c] == int(r["probs"][c] >= r["threshold"]) for c in config.LABEL_COLS))
    check(f"bin_fused = bert | emoji · {tag}",
          all(r["pred_fused"][c] == (r["pred_bert"][c] | r["pred_emoji"][c])
              for c in config.LABEL_COLS))
    check(f"emoji sesuai EMOJI_RX · {tag}", r["emoji"]["found"] == pp.find_emojis(text),
          str(r["emoji"]["found"]))
    check(f"active == label fused · {tag}",
          r["active"] == [c for c in config.LABEL_COLS if r["pred_fused"][c]])
    check(f"jejak pipeline 8 tahap · {tag}", len(r["pipeline"]["steps"]) == 8,
          f"{len(r['pipeline']['steps'])} tahap")
    check(f"wordpiece <= MAX_LEN · {tag}", r["pipeline"]["n_wordpieces"] <= config.MAX_LEN,
          f"{r['pipeline']['n_wordpieces']} token")

# Emoji harus benar-benar mengubah keputusan fusi.
r = M.predict("biasa saja 😭")
check("emoji 😭 memicu label sedih pada jalur emoji", r["pred_emoji"]["sedih"] == 1)
check("😭 membuat sedih aktif di hasil fusi", r["pred_fused"]["sedih"] == 1)

# ---------------------------------------------------------------------------
print("\n[5] Checkpoint tidak ada")
# ---------------------------------------------------------------------------
config.CHECKPOINT_PATH = tmp / "tidak_ada.pt"
st = M.load()
check("status == 'no_checkpoint'", st.status == "no_checkpoint", f"dapat '{st.status}'")
check("loaded == False", st.loaded is False)

# ---------------------------------------------------------------------------
import shutil

shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 68)
if fails:
    print(f"GAGAL — {len(fails)} pemeriksaan tidak lolos:")
    for f in fails:
        print("  ✗", f)
    sys.exit(1)
print("SEMUA PEMERIKSAAN LOLOS")
print("=" * 68)
