"""
SIPEMO — latih IndoBERT-base (baseline) dan simpan checkpoint untuk backend.

Cerminan `train_one()` cell 9 `multilabel_bert_comparison.ipynb`, dijalankan
hanya untuk satu model: `IndoBERT-base (baseline)` — model dengan F1-macro
tertinggi pada cell 11, yaitu model yang dilayani backend.

Hyperparameter, split, penjadwal, sweep threshold, dan isi checkpoint mengikuti
notebook:

    split       train_test_split(seed=42) 80/10/10   -> cell 5
    tokenizer   indobenchmark/indobert-base-p1       -> cell 6
    MAX_LEN 96  BATCH 16  EPOCHS 3  LR 2e-5          -> cell 1
    threshold   sweep np.arange(0.20, 0.55, 0.05)    -> cell 8
    checkpoint  {'state_dict', 'thr', 'f1_macro'}    -> cell 9

`resize_token_embeddings()` sengaja TIDAK dipanggil, sama seperti cell 9.
config.vocab_size IndoBERT-base = 50.000 sedangkan len(tokenizer) = 30.521;
melakukan resize akan menghasilkan checkpoint selebar 30.521 yang ditolak gate
verifikasi backend.

Jalankan:
    .venv\\Scripts\\python training/train_indobert.py
    .venv\\Scripts\\python training/train_indobert.py --benchmark   # ukur dulu
"""

from __future__ import annotations

import argparse
import ast
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "backend"))

import config as cfg  # noqa: E402

CORPUS = PROJECT_DIR / "data" / "labeled_silver.xlsx"
CHECKPOINT = cfg.MODEL_DIR / "best_indobert_base.pt"

# Keluaran cell 5 — split harus persis segini atau ada yang berubah di korpus.
NOTEBOOK_SPLIT = {"train": 1750, "val": 219, "test": 219, "total": 2188}


class TxtDS(Dataset):
    """Salinan `TxtDS` cell 7."""

    def __init__(self, texts, labels, tok, max_len):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.float),
        }


def best_thresh(logits, y_true):
    """
    Salinan `best_thresh()` cell 8, dengan satu perbedaan yang disengaja:
    hasilnya dikembalikan sebagai float Python, bukan np.float64.

    Cell 8 mengembalikan `best_t` dari np.arange() dan `best_f` dari f1_score(),
    keduanya np.float64. Cell 9 menyimpan nilai itu apa adanya ke checkpoint:

        torch.save({'state_dict': ..., 'thr': best_t, 'f1_macro': f}, path)

    torch.load(..., weights_only=True) menolak checkpoint semacam itu dengan
    UnpicklingError 'Unsupported global: numpy._core.multiarray.scalar', karena
    sejak torch 2.6 weights_only default True dan skalar numpy tidak masuk
    daftar aman bawaan. backend/model.py memuat checkpoint persis dengan cara
    itu, jadi checkpoint ber-skalar-numpy akan DITOLAK backend.

    Menyimpan float Python membuat checkpoint bisa dibaca tanpa perlu
    melonggarkan weights_only. backend/model.py tetap dibuat toleran terhadap
    skalar numpy supaya checkpoint hasil notebook juga bisa dipakai.
    """
    best_t, best_f = 0.5, -1
    for t in np.arange(0.20, 0.55, 0.05):
        f = f1_score(y_true, (logits >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), float(best_f)


def evaluate(model, loader, device):
    """Salinan `evaluate()` cell 8."""
    model.eval()
    out_l, out_y = [], []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            mk = b["attention_mask"].to(device)
            out_l.append(
                torch.sigmoid(model(input_ids=ids, attention_mask=mk).logits)
                .cpu()
                .numpy()
            )
            out_y.append(b["labels"].cpu().numpy())
    return np.concatenate(out_l), np.concatenate(out_y)


# Hasil test set IndoBERT-base pada eksekusi notebook (cell 11) — pembanding.
NOTEBOOK_TEST = {
    "F1_micro": 0.993377,
    "F1_macro": 0.994457,
    "F1_samples": 0.995434,
    "Hamming": 0.003425,
    "SubsetAcc": 0.986301,
}


def report_test(model, te, device, best_t, best_v, t_start):
    """Evaluasi test set lalu cetak ringkasannya berdampingan dengan notebook."""
    tl, ty = evaluate(model, te, device)
    yp = (tl >= best_t).astype(int)
    got = {
        "F1_micro": f1_score(ty, yp, average="micro", zero_division=0),
        "F1_macro": f1_score(ty, yp, average="macro", zero_division=0),
        "F1_samples": f1_score(ty, yp, average="samples", zero_division=0),
        "Hamming": hamming_loss(ty, yp),
        "SubsetAcc": accuracy_score(ty, yp),
    }

    print(f"\nhasil test set (thr={best_t:.2f}):")
    print(f"  {'metrik':<12} {'skrip':>9}  {'notebook':>9}")
    for k, v in got.items():
        print(f"  {k:<12} {v:>9.4f}  {NOTEBOOK_TEST[k]:>9.4f}")

    per_label = f1_score(ty, yp, average=None, zero_division=0)
    print("\n  F1 per label:")
    for i, c in enumerate(cfg.LABEL_COLS):
        print(f"    {c:<7}: {per_label[i]:.4f}")

    print(
        f"\ncheckpoint : {CHECKPOINT}"
        f"\n  val F1-macro : {best_v:.4f}  (notebook: 0.9789)"
        f"\n  threshold    : {best_t:.2f}  (notebook: 0.30)"
    )
    if t_start is not None:
        print(f"  durasi       : {(time.time() - t_start) / 60:.1f} menit")


def _numpy_safe_globals():
    """Kelas numpy yang dibutuhkan untuk merekonstruksi skalar di checkpoint."""
    allow = []
    for path in ("numpy._core.multiarray.scalar", "numpy.core.multiarray.scalar"):
        mod, _, attr = path.rpartition(".")
        try:
            allow.append(getattr(__import__(mod, fromlist=[attr]), attr))
        except (ImportError, AttributeError):
            pass
    allow.append(np.dtype)
    allow += [
        getattr(np.dtypes, n)
        for n in ("Float64DType", "Float32DType", "Int64DType")
        if hasattr(np, "dtypes") and hasattr(np.dtypes, n)
    ]
    return allow


def load_checkpoint(path, device):
    """Muat checkpoint; toleran terhadap skalar numpy peninggalan cell 9."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        with torch.serialization.safe_globals(_numpy_safe_globals()):
            return torch.load(path, map_location=device, weights_only=True)


def load_corpus():
    import pandas as pd

    if not CORPUS.exists():
        print(
            f"ERROR: korpus belum ada: {CORPUS}\n"
            "Jalankan dulu: .venv\\Scripts\\python training/build_corpus.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    df = pd.read_excel(CORPUS)
    # to_excel menyimpan list sebagai teks; kembalikan ke list int.
    df["labels"] = df["labels"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else list(v)
    )
    return df


def make_splits(n):
    """Salinan cell 5 — dua kali train_test_split dengan random_state=SEED."""
    idx_all = np.arange(n)
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.2, random_state=cfg.SEED)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.5, random_state=cfg.SEED)
    return idx_train, idx_val, idx_test


def main() -> int:
    p = argparse.ArgumentParser(description="Latih IndoBERT-base untuk SIPEMO.")
    p.add_argument(
        "--benchmark",
        type=int,
        metavar="N",
        help="ukur kecepatan N step lalu berhenti (perkiraan durasi, tanpa melatih)",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="lewati training; evaluasi checkpoint yang sudah ada pada test set",
    )
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    a = p.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)

    # cell 1
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device: {device} | torch: {torch.__version__} | "
        f"threads: {torch.get_num_threads()}"
    )

    df = load_corpus()
    y = np.array(df["labels"].tolist(), dtype=np.float32)
    idx_train, idx_val, idx_test = make_splits(len(df))
    got = {
        "train": len(idx_train),
        "val": len(idx_val),
        "test": len(idx_test),
        "total": len(df),
    }
    print(f"split : {got}")
    if got != NOTEBOOK_SPLIT:
        print(f"  PERINGATAN: split notebook {NOTEBOOK_SPLIT} — korpus berubah?")

    print(f"model : {cfg.SERVED_MODEL_NAME} ({cfg.SERVED_MODEL_ID})")
    tok = AutoTokenizer.from_pretrained(cfg.SERVED_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.SERVED_MODEL_ID,
        num_labels=len(cfg.LABEL_COLS),
        problem_type="multi_label_classification",
    ).to(device)
    print(f"        vocab embedding: {model.get_input_embeddings().weight.shape[0]:,}")

    texts = df["text_norm"].values
    tr = DataLoader(
        TxtDS(texts[idx_train].tolist(), y[idx_train].tolist(), tok, cfg.MAX_LEN),
        batch_size=cfg.BATCH,
        shuffle=True,
    )
    va = DataLoader(
        TxtDS(texts[idx_val].tolist(), y[idx_val].tolist(), tok, cfg.MAX_LEN),
        batch_size=cfg.BATCH * 2,
    )
    te = DataLoader(
        TxtDS(texts[idx_test].tolist(), y[idx_test].tolist(), tok, cfg.MAX_LEN),
        batch_size=cfg.BATCH * 2,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    sch = get_linear_schedule_with_warmup(opt, 0, len(tr) * a.epochs)

    # ---- benchmark -------------------------------------------------------
    if a.benchmark:
        model.train()
        t0 = time.time()
        for i, b in enumerate(tr):
            if i >= a.benchmark:
                break
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device),
            )
            out.loss.backward()
            opt.step()
            sch.step()
            opt.zero_grad()
        per_step = (time.time() - t0) / a.benchmark
        total = per_step * len(tr) * a.epochs
        print(
            f"\nbenchmark: {per_step:.2f} s/step | {len(tr)} step/epoch x "
            f"{a.epochs} epoch = {len(tr) * a.epochs} step"
            f"\nperkiraan training: {total / 60:.0f} menit (+ evaluasi)"
        )
        return 0

    # ---- evaluasi checkpoint yang sudah ada ------------------------------
    if a.eval_only:
        if not CHECKPOINT.exists():
            print(f"ERROR: belum ada checkpoint di {CHECKPOINT}", file=sys.stderr)
            return 1
        ck = load_checkpoint(CHECKPOINT, device)
        model.load_state_dict(ck["state_dict"], strict=True)
        best_t = float(ck.get("thr", cfg.DEFAULT_THRESHOLD))
        best_v = float(ck.get("f1_macro", float("nan")))
        print(f"muat  : {CHECKPOINT.name} (thr={best_t:.2f} val_F1={best_v:.4f})")

        # Checkpoint dari cell 9 menyimpan thr/f1_macro sebagai np.float64,
        # yang ditolak torch.load(weights_only=True) — termasuk oleh backend.
        # Tulis ulang metadatanya sebagai float Python; bobotnya tidak disentuh.
        if not all(type(ck.get(k)) is float for k in ("thr", "f1_macro")):
            torch.save(
                {"state_dict": ck["state_dict"], "thr": best_t, "f1_macro": best_v},
                CHECKPOINT,
            )
            print(
                "        metadata skalar numpy ditulis ulang sebagai float Python "
                "(bobot tidak berubah)"
            )
        report_test(model, te, device, best_t, best_v, None)
        return 0

    # ---- training (mengikuti train_one() cell 9) -------------------------
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best_t, best_v = 0.5, -1
    t_start = time.time()

    for ep in range(a.epochs):
        model.train()
        losses = []
        t_ep = time.time()
        for i, b in enumerate(tr):
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device),
            )
            out.loss.backward()
            opt.step()
            sch.step()
            opt.zero_grad()
            losses.append(out.loss.item())
            if (i + 1) % 20 == 0:
                done = i + 1
                eta = (time.time() - t_ep) / done * (len(tr) - done)
                print(
                    f"  ep{ep + 1} step {done}/{len(tr)} "
                    f"loss={np.mean(losses[-20:]):.4f} eta_epoch={eta / 60:.1f}m",
                    flush=True,
                )

        vl, vy = evaluate(model, va, device)
        t, f = best_thresh(vl, vy)
        print(
            f"  [{cfg.SERVED_MODEL_NAME}] ep={ep + 1} loss={np.mean(losses):.4f} "
            f"val_F1macro={f:.4f} thr={t:.2f} ({(time.time() - t_ep) / 60:.1f}m)",
            flush=True,
        )
        if f > best_v:
            best_v, best_t = f, t
            torch.save(
                {"state_dict": model.state_dict(), "thr": best_t, "f1_macro": f},
                CHECKPOINT,
            )
            print(f"        checkpoint disimpan (val F1-macro {f:.4f})", flush=True)

    # ---- evaluasi test dengan bobot terbaik ------------------------------
    best_ckpt = load_checkpoint(CHECKPOINT, device)
    model.load_state_dict(best_ckpt["state_dict"], strict=True)
    report_test(model, te, device, best_t, best_v, t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
