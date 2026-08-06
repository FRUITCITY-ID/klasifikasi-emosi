"""
SIPEMO — pemuatan model & inferensi.

Alur inferensi identik dengan `predict_one()` pada notebook cell 15:

    norm      = SLANG(_norm(raw))
    prob      = sigmoid(model(tokenizer(norm)).logits)
    bin_bert  = prob >= threshold
    bin_emoji = leksikon_emoji(EMOJI_RX.findall(raw))
    bin_fused = bin_bert | bin_emoji        <- fusi level keputusan (OR bitwise)

Perbedaan yang disengaja dengan notebook — dan alasannya:

    Notebook versi awal menulis SEMUA model ke satu file `best_bert.pt`, sehingga
    file terakhir di disk adalah bobot model yang dilatih paling akhir
    (mBERT-base), bukan model dengan F1 tertinggi. Cell 15 kemudian mencetak
    nama & metrik "IndoBERT-base (baseline)" di atas prediksi yang sebenarnya
    dihasilkan bobot mBERT.

    Backend ini menolak melakukan hal itu. Checkpoint diverifikasi dulu
    terhadap arsitektur IndoBERT-base; kalau tidak cocok, model TIDAK dimuat
    dan API melaporkan statusnya secara eksplisit. Lebih baik menampilkan
    "model belum siap" daripada menyajikan angka yang keliru.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import config
import preprocessing as pp

# numpy/torch/transformers sengaja TIDAK diimpor di level modul. Kalau salah
# satunya belum terpasang, server harus tetap hidup dan melaporkannya lewat
# /api/health — bukan gagal start sehingga seluruh UI ikut mati.

# Ukuran matriks embedding tiap model saat DILATIH (cell 9 tidak memanggil
# resize_token_embeddings, jadi bentuknya mengikuti config.vocab_size, bukan
# len(tokenizer)). Dipakai hanya untuk pesan diagnostik.
VOCAB_HINTS: dict[int, str] = {
    50000: "IndoBERT-base-p1 atau IndoBERT-base-p2 (IndoNLU)",
    31923: "IndoBERTweet atau IndoBERT-IndoLEM",
    119547: "bert-base-multilingual-cased (mBERT-base)",
    250002: "xlm-roberta-base",
}

# Val F1-macro terbaik tiap model saat training (output cell 9). Notebook
# menyimpan nilai ini ke checkpoint sebagai key 'f1_macro', jadi bisa dipakai
# sebagai sidik jari sekunder untuk memastikan checkpoint milik model yang benar.
VAL_F1_FINGERPRINT: dict[str, float] = {
    "IndoBERT-base (baseline)": 0.9789,
    "IndoBERT-IndoNLU": 0.9880,
    "IndoBERTweet": 0.8658,
    "mBERT-base": 0.7816,
    "IndoBERT-IndoLEM": 0.7582,
    "XLM-RoBERTa-base": 0.6746,
}


@dataclass
class ModelState:
    """Status pemuatan model, dilaporkan apa adanya lewat /api/health."""

    loaded: bool = False
    status: str = "not_initialized"   # ready | no_checkpoint | mismatch | error
    message: str = ""
    threshold: float = config.DEFAULT_THRESHOLD
    threshold_source: str = "default (notebook cell 11)"
    device: str = "cpu"
    checkpoint_path: str = str(config.CHECKPOINT_PATH)
    checkpoint_found: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "status": self.status,
            "message": self.message,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_found": self.checkpoint_found,
            "warnings": self.warnings,
            "model_name": config.SERVED_MODEL_NAME,
            "model_id": config.SERVED_MODEL_ID,
        }


STATE = ModelState()
_tokenizer = None
_model = None
_torch = None
_np = None


def _numpy_safe_globals(numpy: Any) -> list:
    """
    Kelas numpy yang diperlukan untuk merekonstruksi skalar di checkpoint.

    Cell 8 mengembalikan threshold dari `np.arange()` dan F1 dari `f1_score()`,
    keduanya np.float64, lalu cell 9 menyimpannya apa adanya:

        torch.save({'state_dict': ..., 'thr': best_t, 'f1_macro': f}, path)

    Sejak torch 2.6 `weights_only` default True dan skalar numpy tidak termasuk
    daftar aman bawaan, sehingga checkpoint hasil notebook ditolak dengan
    UnpicklingError. Melonggarkan ke weights_only=False bukan pilihan — itu
    membuka eksekusi pickle sembarang. Yang dilakukan di sini adalah
    mengizinkan tepat kelas-kelas numpy yang dibutuhkan; rekonstruksinya hanya
    membentuk skalar, bukan memanggil kode arbitrer.
    """
    allow: list = []
    for path in ("numpy._core.multiarray.scalar", "numpy.core.multiarray.scalar"):
        mod, _, attr = path.rpartition(".")
        try:
            allow.append(getattr(__import__(mod, fromlist=[attr]), attr))
        except (ImportError, AttributeError):
            pass
    allow.append(numpy.dtype)
    if hasattr(numpy, "dtypes"):
        allow += [
            getattr(numpy.dtypes, n)
            for n in ("Float64DType", "Float32DType", "Int64DType")
            if hasattr(numpy.dtypes, n)
        ]
    return allow


def _extract_state_dict(raw: Any) -> dict:
    """Ambil state_dict dari checkpoint & buang prefix 'module.' (cell 15)."""
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _checkpoint_vocab_size(sd: dict) -> int | None:
    for key in (
        "bert.embeddings.word_embeddings.weight",
        "roberta.embeddings.word_embeddings.weight",
    ):
        if key in sd:
            return int(sd[key].shape[0])
    return None


def load() -> ModelState:
    """Muat tokenizer + model + checkpoint. Aman dipanggil berulang."""
    global _tokenizer, _model, _torch, _np, STATE

    STATE = ModelState()

    try:
        import numpy
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        _torch = torch
        _np = numpy
    except ImportError as e:
        STATE.status = "error"
        STATE.message = (
            f"Dependensi belum terpasang ({e.name}). Jalankan: "
            "pip install -r backend/requirements.txt"
        )
        return STATE

    device = "cuda" if (torch.cuda.is_available() and not config.FORCE_CPU) else "cpu"
    STATE.device = device

    # ---- tokenizer + arsitektur dasar ------------------------------------
    try:
        _tokenizer = AutoTokenizer.from_pretrained(config.SERVED_MODEL_ID)
        _model = AutoModelForSequenceClassification.from_pretrained(
            config.SERVED_MODEL_ID,
            num_labels=len(config.LABEL_COLS),
            problem_type="multi_label_classification",
        ).to(device)

        # Sengaja TIDAK memanggil resize_token_embeddings().
        #
        # Cell 15 memanggilnya dengan len(tokenizer), tapi cell 9 — yang
        # menghasilkan bobotnya — tidak. Untuk IndoBERT-base kedua angka itu
        # berbeda: config.vocab_size = 50.000 sedangkan len(tokenizer) = 30.521.
        # Melakukan resize akan menyusutkan embedding ke 30.521 sehingga tidak
        # akan pernah cocok dengan checkpoint hasil training yang lebarnya
        # 50.000 — load_state_dict akan gagal dengan size mismatch.
        #
        # Bentuk bobot ditentukan oleh training, jadi inferensi harus mengikuti
        # cell 9, bukan cell 15.
    except Exception as e:  # noqa: BLE001 — mau menampilkan pesan apa adanya
        STATE.status = "error"
        STATE.message = (
            f"Gagal memuat {config.SERVED_MODEL_ID} dari Hugging Face: "
            f"{type(e).__name__}: {e}"
        )
        return STATE

    # ---- checkpoint -------------------------------------------------------
    ckpt = config.CHECKPOINT_PATH
    STATE.checkpoint_found = ckpt.exists()

    if not STATE.checkpoint_found:
        STATE.status = "no_checkpoint"
        STATE.message = (
            f"Checkpoint IndoBERT-base belum tersedia di {ckpt}. Salin file "
            f"best_indobert_base.pt hasil training ke {config.MODEL_DIR}, lalu "
            "klik 'Muat ulang model'. Jika file disimpan di lokasi lain, set "
            "SIPEMO_CHECKPOINT ke path tersebut sebelum menjalankan SIPEMO. "
            "Prediksi tetap dinonaktifkan agar bobot acak tidak menghasilkan keluaran keliru."
        )
        return STATE

    try:
        # Checkpoint SIPEMO hanya membutuhkan tensor state_dict dan metadata
        # primitif. Mode aman mencegah eksekusi objek pickle arbitrer.
        raw = torch.load(ckpt, map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001
        # Kemungkinan besar checkpoint hasil notebook, yang menyimpan thr &
        # f1_macro sebagai np.float64. Coba lagi dengan skalar numpy
        # diizinkan — weights_only TETAP True, jadi pickle sembarang tetap
        # ditolak. Lihat _numpy_safe_globals().
        try:
            with torch.serialization.safe_globals(_numpy_safe_globals(numpy)):
                raw = torch.load(ckpt, map_location=device, weights_only=True)
            STATE.warnings.append(
                "Checkpoint menyimpan 'thr'/'f1_macro' sebagai skalar numpy "
                "(bawaan cell 8-9 notebook). Berhasil dibaca dengan daftar aman "
                "numpy. Checkpoint dari training/train_indobert.py memakai float "
                "Python sehingga tidak perlu kelonggaran ini."
            )
        except Exception as e:  # noqa: BLE001
            STATE.status = "error"
            STATE.message = f"Gagal membaca checkpoint: {type(e).__name__}: {e}"
            return STATE

    sd = _extract_state_dict(raw)

    # ---- gate 1: ukuran vocab harus cocok ---------------------------------
    ckpt_vocab = _checkpoint_vocab_size(sd)
    model_vocab = int(_model.get_input_embeddings().weight.shape[0])

    if ckpt_vocab is None:
        STATE.warnings.append(
            "Tidak menemukan word_embeddings di checkpoint — verifikasi arsitektur dilewati."
        )
    elif ckpt_vocab != model_vocab:
        hint = VOCAB_HINTS.get(ckpt_vocab, "model tak dikenal")
        detail = (
            f"Checkpoint ini punya vocab {ckpt_vocab:,} sedangkan "
            f"{config.SERVED_MODEL_NAME} ({config.SERVED_MODEL_ID}) butuh "
            f"{model_vocab:,}. Isinya kemungkinan besar {hint}.\n\n"
            "Checkpoint ini kemungkinan dari notebook versi lama: train_one() menyimpan ke "
            "path yang sama untuk setiap model, sehingga best_bert.pt berisi "
            "bobot model YANG DILATIH TERAKHIR (mBERT-base), bukan model dengan "
            "F1 tertinggi.\n\n"
            "Perbaikan: latih ulang IndoBERT-base saja lalu simpan ke file "
            "tersendiri (lihat README bagian 'Menyiapkan checkpoint'), atau "
            "arahkan SIPEMO_CHECKPOINT ke checkpoint IndoBERT-base yang benar."
        )
        if not config.ALLOW_MISMATCH:
            STATE.status = "mismatch"
            STATE.message = detail
            return STATE
        STATE.warnings.append("SIPEMO_ALLOW_MISMATCH=1 — " + detail)

    # ---- gate 2: sidik jari F1 (sekunder, hanya peringatan) ---------------
    ckpt_f1 = raw.get("f1_macro") if isinstance(raw, dict) else None
    if isinstance(ckpt_f1, (int, float)):
        expected = VAL_F1_FINGERPRINT[config.SERVED_MODEL_NAME]
        if abs(float(ckpt_f1) - expected) > 1e-3:
            closest = min(
                VAL_F1_FINGERPRINT.items(), key=lambda kv: abs(kv[1] - float(ckpt_f1))
            )
            STATE.warnings.append(
                f"Val F1-macro pada checkpoint = {float(ckpt_f1):.4f}, sedangkan "
                f"{config.SERVED_MODEL_NAME} seharusnya {expected:.4f}. Nilai ini "
                f"paling dekat dengan '{closest[0]}' ({closest[1]:.4f}). "
                "Pastikan checkpoint-nya benar."
            )

    # ---- muat bobot -------------------------------------------------------
    try:
        result = _model.load_state_dict(sd, strict=False)
    except Exception as e:  # noqa: BLE001
        STATE.status = "mismatch"
        STATE.message = f"Bentuk bobot tidak cocok: {type(e).__name__}: {e}"
        return STATE

    if result.missing_keys:
        STATE.warnings.append(
            f"{len(result.missing_keys)} key hilang saat load "
            f"(contoh: {result.missing_keys[:3]})"
        )
    if result.unexpected_keys:
        STATE.warnings.append(
            f"{len(result.unexpected_keys)} key tak terduga "
            f"(contoh: {result.unexpected_keys[:3]})"
        )

    # ---- threshold --------------------------------------------------------
    if isinstance(raw, dict) and "thr" in raw:
        STATE.threshold = float(raw["thr"])
        STATE.threshold_source = "checkpoint (key 'thr')"
    else:
        STATE.threshold = config.DEFAULT_THRESHOLD
        STATE.threshold_source = "default notebook cell 11 (0.30)"

    _model.eval()
    STATE.loaded = True
    STATE.status = "ready"
    STATE.message = f"{config.SERVED_MODEL_NAME} siap pada {device}."
    return STATE


def predict(raw_text: str) -> dict[str, Any]:
    """Inferensi satu teks, lengkap dengan jejak pipeline untuk UI."""
    if not STATE.loaded or _model is None or _tokenizer is None:
        raise RuntimeError(STATE.message or "Model belum dimuat.")

    t0 = time.perf_counter()
    labels = config.LABEL_COLS

    # --- jalur teks -------------------------------------------------------
    norm, steps, slang_changes = pp.normalize_traced(raw_text)

    enc = _tokenizer(
        norm,
        truncation=True,
        max_length=config.MAX_LEN,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]
    pieces = _tokenizer.convert_ids_to_tokens(input_ids)

    enc = {k: v.to(STATE.device) for k, v in enc.items()}
    with _torch.no_grad():
        logits = _model(**enc).logits.cpu().numpy()[0]

    prob = 1.0 / (1.0 + _np.exp(-logits))
    thr = STATE.threshold
    bin_bert = (prob >= thr).astype(int)

    # --- jalur emoji ------------------------------------------------------
    emojis = pp.find_emojis(raw_text)
    bin_emoji = _np.array(pp.label_emoji_set(emojis), dtype=int)

    # --- fusi level keputusan --------------------------------------------
    bin_fused = (bin_bert | bin_emoji).astype(int)

    n_word_tokens = len(norm.split())
    return {
        "text": raw_text,
        "norm": norm,
        "threshold": thr,
        "labels": labels,
        "probs": {c: float(prob[i]) for i, c in enumerate(labels)},
        "logits": {c: float(logits[i]) for i, c in enumerate(labels)},
        "pred_bert": {c: int(bin_bert[i]) for i, c in enumerate(labels)},
        "pred_emoji": {c: int(bin_emoji[i]) for i, c in enumerate(labels)},
        "pred_fused": {c: int(bin_fused[i]) for i, c in enumerate(labels)},
        "active": [c for i, c in enumerate(labels) if bin_fused[i]],
        "pipeline": {
            "steps": steps,
            "slang_changes": slang_changes,
            "words": norm.split(),
            "n_words": n_word_tokens,
            "wordpieces": pieces,
            "n_wordpieces": len(pieces),
            "max_len": config.MAX_LEN,
            "truncated": len(pieces) >= config.MAX_LEN,
        },
        "emoji": {
            "found": emojis,
            "matches": pp.emoji_matches(emojis),
            "vector": {c: int(bin_emoji[i]) for i, c in enumerate(labels)},
        },
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
