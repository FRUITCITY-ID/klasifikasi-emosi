"""
SIPEMO — buktikan checkpoint lokal setara dengan hasil penelitian di notebook.

Skrip ini TIDAK memakai angka yang diketik ulang oleh siapa pun. Semua pembanding
dibaca langsung dari output tersimpan di `multilabel_bert_comparison.ipynb`
(cell 3, 4, 5, 9), lalu dibandingkan dengan hasil yang dihitung ulang dari
`data/labeled_silver.xlsx` dan `models/best_indobert_base.pt`.

Enam lapis pemeriksaan, makin ke bawah makin ketat:

  1. korpus      — jumlah baris & sebaran label            (cell 3)
  2. statistik   — kardinalitas presisi penuh, kombo, panjang teks  (cell 4)
  3. split       — ukuran train/val/test                   (cell 5)
  4. ground truth— y_true_test 219x4, elemen per elemen    (cell 9)  <- kunci
  5. prediksi    — y_pred_test 219x4, elemen per elemen    (cell 9)  <- kunci
  6. metrik      — F1/Hamming/SubsetAcc presisi penuh       (cell 9)

Lapis 4 yang paling menentukan. Kalau 876 nilai ground-truth cocok semua, maka
korpus, urutan baris, DAN split-nya identik — bukan sekadar totalnya kebetulan
sama. Lapis 5 memeriksa bobot: 876 keputusan biner dari model lokal harus sama
dengan keputusan model yang dilaporkan di skripsi.

Jalankan:
    .venv\\Scripts\\python training/verify_reproduction.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "backend"))
sys.path.insert(0, str(PROJECT_DIR / "training"))

import config as cfg  # noqa: E402
from preprocessing import EMOJI_RX  # noqa: E402
from train_indobert import (  # noqa: E402
    CHECKPOINT,
    TxtDS,
    evaluate,
    load_checkpoint,
    load_corpus,
    make_splits,
)

NOTEBOOK = PROJECT_DIR / "multilabel_bert_comparison.ipynb"
MODEL_NAME = "IndoBERT-base (baseline)"

results: list[tuple[bool, str, str, str]] = []


def check(ok: bool, name: str, got, want) -> None:
    results.append((bool(ok), name, str(got), str(want)))


# ---------------------------------------------------------------------------
# Baca output notebook
# ---------------------------------------------------------------------------
def notebook_streams() -> dict[int, str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        txt = ""
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                txt += "".join(o.get("text", []))
            elif o.get("output_type") == "execute_result":
                txt += "".join(o.get("data", {}).get("text/plain", []))
        out[i] = txt
    return out


def parse_notebook(streams: dict[int, str]) -> dict:
    """Ambil angka pembanding dari teks output, lewat regex — bukan hafalan."""
    nb: dict = {}
    c3, c4, c5, c9 = streams[3], streams[4], streams[5], streams[9]

    m = re.search(r"rows after prep:\s*(\d+)\s*\|\s*emoji rate:\s*([\d.]+)%", c3)
    nb["rows_prep"], nb["emoji_rate"] = int(m.group(1)), float(m.group(2))
    nb["rows_labeled"] = int(re.search(r"\|\s*rows=(\d+)\s*\(", c3).group(1))
    nb["label_counts"] = {
        c: int(re.search(rf"^\s+{c}:\s*(\d+)/", c3, re.M).group(1))
        for c in cfg.LABEL_COLS
    }

    m = re.search(r"cardinality mean:\s*([\d.]+)\s*max:\s*(\d+)", c4)
    nb["card_mean"], nb["card_max"] = float(m.group(1)), int(m.group(2))
    combos = re.search(r"top combo:\s*(\[.*?\])\s*\n", c4, re.S).group(1)
    # 'np.int32(1)' -> '1' supaya bisa di-literal_eval
    nb["top_combos"] = [
        (tuple(t), n)
        for t, n in ast.literal_eval(re.sub(r"np\.int32\((\d+)\)", r"\1", combos))
    ]
    m = re.search(r"text len p50/p95:\s*(\d+)\s+(\d+)", c4)
    nb["p50"], nb["p95"] = int(m.group(1)), int(m.group(2))

    nb["split"] = ast.literal_eval(re.search(r"\{'train':.*?\}", c5).group(0))

    nb["epochs"] = [
        {"ep": int(e), "loss": float(l), "val_f1": float(f), "thr": float(t)}
        for e, l, f, t in re.findall(
            rf"\[{re.escape(MODEL_NAME)}\] ep=(\d+) loss=([\d.]+) "
            r"val_F1macro=([\d.]+) thr=([\d.]+)",
            c9,
        )
    ]

    i = c9.find(f"{{'model': '{MODEL_NAME}'")
    nb["result"] = ast.literal_eval(c9[i : c9.find("\n", i)])
    return nb


# ---------------------------------------------------------------------------
def main() -> int:
    if not NOTEBOOK.exists():
        print(f"ERROR: notebook tidak ditemukan: {NOTEBOOK}", file=sys.stderr)
        return 1
    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint tidak ditemukan: {CHECKPOINT}", file=sys.stderr)
        return 1

    print("pembanding dibaca dari output tersimpan di", NOTEBOOK.name)
    nb = parse_notebook(notebook_streams())

    # ---- 1. korpus -------------------------------------------------------
    # 'rows after prep' dan 'emoji rate' dicetak cell 3 pada frame SEBELUM
    # filter pelabelan (19.491 baris), bukan pada korpus berlabel (2.188).
    # Keduanya harus diperiksa pada populasi itu.
    print("\n[1] Korpus (cell 3)")
    from build_corpus import DEFAULT_DATA, prepare

    prep = prepare(DEFAULT_DATA)
    check(len(prep) == nb["rows_prep"], "baris setelah prep", len(prep), nb["rows_prep"])
    rate = round(prep["emoji_count"].gt(0).mean() * 100, 1)
    check(rate == nb["emoji_rate"], "emoji rate (pada frame prep)", rate, nb["emoji_rate"])

    df = load_corpus()
    y = np.array(df["labels"].tolist(), dtype=np.float32)
    check(len(df) == nb["rows_labeled"], "baris berlabel", len(df), nb["rows_labeled"])
    for i, c in enumerate(cfg.LABEL_COLS):
        got = int(y[:, i].sum())
        check(got == nb["label_counts"][c], f"jumlah label {c}", got, nb["label_counts"][c])

    # ---- 2. statistik dataset -------------------------------------------
    print("[2] Statistik dataset (cell 4)")
    # Cell 4 menghitung dari array int32: y_preview.sum(1).mean(). Memakai
    # float32 di sini akan meleset di digit ke-9 karena akumulasi presisi
    # tunggal, bukan karena datanya berbeda — jadi dtype-nya harus disamakan.
    y_i32 = np.array(df["labels"].tolist(), dtype=np.int32)
    npos = y_i32.sum(1)
    card_mean = float(npos.mean())
    check(
        repr(card_mean) == repr(nb["card_mean"]),
        "kardinalitas rata-rata (presisi penuh)",
        repr(card_mean),
        repr(nb["card_mean"]),
    )
    check(int(npos.max()) == nb["card_max"], "kardinalitas maks", int(npos.max()), nb["card_max"])

    from collections import Counter

    got_combos = Counter(map(tuple, y_i32)).most_common(5)
    got_combos = [(tuple(int(x) for x in t), n) for t, n in got_combos]
    check(got_combos == nb["top_combos"], "5 kombinasi label teratas", got_combos, nb["top_combos"])

    lens = df["text_norm"].str.split().str.len()
    check(int(lens.quantile(0.5)) == nb["p50"], "panjang teks p50", int(lens.quantile(0.5)), nb["p50"])
    check(int(lens.quantile(0.95)) == nb["p95"], "panjang teks p95", int(lens.quantile(0.95)), nb["p95"])

    # ---- 2b. angka turunan yang ditampilkan UI --------------------------
    # lexicon_top_triggers tidak ada di output notebook — dihitung dari korpus.
    # Karena tetap tampil di tab Dataset & Metode, nilainya harus bisa dihitung
    # ulang, bukan dipercaya begitu saja.
    print("[2b] Kata pemicu leksikon (dihitung ulang dari korpus)")
    import research as R
    from collections import Counter

    word_rx = re.compile(r"[\w]+")
    from preprocessing import LEX

    tally = {c: Counter() for c in cfg.LABEL_COLS}
    for t in df["text_norm"]:
        ws = set(word_rx.findall(t.lower()))
        for c in cfg.LABEL_COLS:
            for k in ws & LEX[c]:
                tally[c][k] += 1
    for c in cfg.LABEL_COLS:
        got = [[w, n] for w, n in tally[c].most_common(4)]
        check(got == R.DATASET["lexicon_top_triggers"][c], f"pemicu teratas {c}", got,
              R.DATASET["lexicon_top_triggers"][c])

    # Entri emoji di LEX tidak boleh pernah cocok saat pelabelan.
    n_emoji_hits = sum(
        len((set(word_rx.findall(t.lower())) | set(EMOJI_RX.findall(t)))
            & {k for k in LEX[c] if not k.isascii()})
        for t in df["text_norm"]
        for c in cfg.LABEL_COLS
    )
    check(n_emoji_hits == 0, "entri emoji leksikon tidak aktif saat pelabelan",
          n_emoji_hits, 0)

    # ---- 3. split --------------------------------------------------------
    print("[3] Split (cell 5)")
    idx_train, idx_val, idx_test = make_splits(len(df))
    got_split = {
        "train": len(idx_train),
        "val": len(idx_val),
        "test": len(idx_test),
        "total": len(df),
    }
    check(got_split == nb["split"], "ukuran split", got_split, nb["split"])

    # ---- 4. ground truth test set, elemen per elemen --------------------
    print("[4] Ground truth test set — 219x4 elemen (cell 9)")
    nb_true = np.array(nb["result"]["y_true_test"], dtype=np.float32)
    got_true = y[idx_test]
    check(got_true.shape == nb_true.shape, "bentuk y_true_test", got_true.shape, nb_true.shape)
    n_same = int((got_true == nb_true).sum())
    check(
        np.array_equal(got_true, nb_true),
        f"y_true_test identik ({got_true.size} nilai)",
        f"{n_same}/{got_true.size} cocok",
        "semua cocok",
    )

    # ---- 5. prediksi model lokal vs notebook ----------------------------
    print("[5] Prediksi model — menjalankan inferensi ulang di test set")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.SERVED_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.SERVED_MODEL_ID,
        num_labels=len(cfg.LABEL_COLS),
        problem_type="multi_label_classification",
    ).to(device)
    ck = load_checkpoint(CHECKPOINT, device)
    model.load_state_dict(ck["state_dict"], strict=True)
    thr = float(ck["thr"])
    check(thr == float(nb["result"]["threshold"]), "threshold", thr, nb["result"]["threshold"])

    te = DataLoader(
        TxtDS(
            df["text_norm"].values[idx_test].tolist(),
            y[idx_test].tolist(),
            tok,
            cfg.MAX_LEN,
        ),
        batch_size=cfg.BATCH * 2,
    )
    tl, ty = evaluate(model, te, device)
    yp = (tl >= thr).astype(int)
    nb_pred = np.array(nb["result"]["y_pred_test"], dtype=int)
    n_same = int((yp == nb_pred).sum())
    check(
        np.array_equal(yp, nb_pred),
        f"y_pred_test identik ({yp.size} keputusan)",
        f"{n_same}/{yp.size} cocok",
        "semua cocok",
    )

    # ---- 6. metrik presisi penuh ----------------------------------------
    print("[6] Metrik test set — presisi penuh (cell 9)")
    got_metrics = {
        "F1_micro": f1_score(ty, yp, average="micro", zero_division=0),
        "F1_macro": f1_score(ty, yp, average="macro", zero_division=0),
        "F1_samples": f1_score(ty, yp, average="samples", zero_division=0),
        "Hamming": hamming_loss(ty, yp),
        "SubsetAcc": accuracy_score(ty, yp),
    }
    for k, v in got_metrics.items():
        want = float(nb["result"][k])
        check(repr(float(v)) == repr(want), f"{k} (presisi penuh)", repr(float(v)), repr(want))

    per_label = [float(x) for x in f1_score(ty, yp, average=None, zero_division=0)]
    want_pl = [float(x) for x in nb["result"]["per_label_F1"]]
    for i, c in enumerate(cfg.LABEL_COLS):
        check(
            repr(per_label[i]) == repr(want_pl[i]),
            f"F1 {c} (presisi penuh)",
            repr(per_label[i]),
            repr(want_pl[i]),
        )

    # ---- 7. jejak training ----------------------------------------------
    log = PROJECT_DIR / "training" / "train.log"
    if log.exists():
        print("[7] Jejak training per epoch (cell 9 vs training/train.log)")
        mine = re.findall(
            rf"\[{re.escape(MODEL_NAME)}\] ep=(\d+) loss=([\d.]+) "
            r"val_F1macro=([\d.]+) thr=([\d.]+)",
            log.read_text(encoding="utf-8", errors="ignore"),
        )
        for e in nb["epochs"]:
            row = next((r for r in mine if int(r[0]) == e["ep"]), None)
            if row is None:
                check(False, f"epoch {e['ep']} ada di log lokal", "tidak ada", "ada")
                continue
            got = (float(row[1]), float(row[2]), float(row[3]))
            want = (e["loss"], e["val_f1"], e["thr"])
            check(
                got == want,
                f"epoch {e['ep']} loss/val_F1/thr (4 desimal)",
                got,
                want,
            )
    else:
        print("[7] training/train.log tidak ada — jejak per epoch dilewati")

    # ---- ringkasan -------------------------------------------------------
    width = max(len(n) for _, n, _, _ in results) + 2
    print("\n" + "=" * 78)
    fails = [r for r in results if not r[0]]
    for ok, name, got, want in results:
        mark = "OK  " if ok else "BEDA"
        line = f"  {mark}  {name:<{width}}"
        print(line if ok else f"{line}\n        lokal   : {got}\n        notebook: {want}")
    print("=" * 78)
    print(
        f"{len(results) - len(fails)}/{len(results)} pemeriksaan cocok"
        + ("" if fails else " — hasil lokal setara dengan hasil penelitian di notebook.")
    )
    if fails:
        print("\nYang tidak cocok:")
        for _, name, got, want in fails:
            print(f"  - {name}: lokal={got} notebook={want}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
