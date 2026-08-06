"""
SIPEMO — bangun korpus berlabel silver dari XLSX mentah.

Reproduksi cell 3 `multilabel_bert_comparison.ipynb`. Keluarannya
`data/labeled_silver.xlsx` — masukan untuk `train_indobert.py`.

Fungsi preprocessing TIDAK disalin ulang di sini; semuanya diimpor dari
`backend/preprocessing.py`, yang sudah diuji identik dengan notebook lewat
`backend/test_pipeline.py`. Dengan begitu teks yang dipakai saat training
dijamin sama persis dengan teks yang dipakai backend saat inferensi.

--------------------------------------------------------------------------
Soal jalur pelabelan
--------------------------------------------------------------------------
Cell 3 mencoba melabeli lewat `StevenLimcorn/indonesian-roberta-base-emotion-classifier`
lalu mundur ke leksikon kalau gagal:

    _HF_MAP = {'Senang':0,'Sedih':1,'Marah':2,'Takut':3}
    def _hf_one(text):
        ...
        for r in s:
            i = _HF_MAP.get(r['label'])
            if i is not None and r['score'] >= 0.30: bins[i] = 1
        return bins

    def _label_row(text):
        if _pipe is not None:
            try:
                b = _hf_one(text)
                if sum(b) > 0: return b      # <-- tidak pernah tercapai
            except Exception:
                pass
        return _lex_one(text)

Model itu mengeluarkan `sadness / anger / love / fear / happy` (lihat
config.json-nya), sedangkan `_HF_MAP` berkunci `Senang / Sedih / Marah / Takut`.
Tidak ada kunci yang cocok, sehingga `_HF_MAP.get(...)` selalu None, `bins`
tetap `[0,0,0,0]`, `sum(b) > 0` selalu False, dan SETIAP baris memakai
`_lex_one`. Notebook tetap mencetak `mode=HF top-k=2` karena penanda itu hanya
mengecek apakah pipeline berhasil dimuat, bukan apakah keluarannya dipakai.

Skrip ini melewatkan pemanggilan model tersebut — hasilnya identik dan hemat
~54 menit. Kesetaraan dibuktikan angka, bukan klaim: `--verify` membandingkan
jumlah baris dan sebaran label dengan keluaran notebook yang tercatat.

Untuk membuktikan sendiri bahwa jalur HF memang mati:

    python training/build_corpus.py --probe-hf-mapping
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "backend"))

from config import LABEL_COLS  # noqa: E402
from preprocessing import EMOJI_RX, LEX, _norm, apply_slang  # noqa: E402

DEFAULT_DATA = PROJECT_DIR / "tugas-akhir_cleaned.xlsx"
DEFAULT_OUT = PROJECT_DIR / "data" / "labeled_silver.xlsx"

# Keluaran cell 3 pada eksekusi notebook yang menghasilkan metrik di
# backend/research.py. Dipakai sebagai uji regresi, bukan sekadar catatan.
NOTEBOOK_ROWS_AFTER_PREP = 19491
NOTEBOOK_ROWS_LABELED = 2188
NOTEBOOK_LABEL_COUNTS = {"senang": 1492, "sedih": 297, "marah": 170, "takut": 333}
NOTEBOOK_EMOJI_RATE = 26.3  # persen baris yang memuat emoji, setelah prep

WORD_RX = re.compile(r"[\w]+")


def lex_one(text: str) -> list[int]:
    """Salinan `_lex_one()` cell 3 — leksikon diimpor dari backend."""
    ws = set(WORD_RX.findall(text.lower())) | set(EMOJI_RX.findall(text))
    return [int(any(k in ws for k in LEX[c])) for c in LABEL_COLS]


def probe_hf_mapping() -> int:
    """Jalankan model pelabel pada beberapa kalimat, tunjukkan labelnya."""
    from transformers import pipeline

    print("memuat StevenLimcorn/indonesian-roberta-base-emotion-classifier ...")
    pipe = pipeline(
        "text-classification",
        model="StevenLimcorn/indonesian-roberta-base-emotion-classifier",
        device=-1,
        truncation=True,
        max_length=96,
    )
    hf_map = {"Senang": 0, "Sedih": 1, "Marah": 2, "Takut": 3}
    samples = [
        "aku sangat bahagia hari ini",
        "hatiku hancur dan aku menangis semalaman",
        "aku benci sekali dengan sikapnya",
        "aku takut sekali menghadapi besok",
    ]
    print(f"\n_HF_MAP di cell 3 = {hf_map}\n")
    n_mapped = 0
    for s in samples:
        out = pipe(s)
        out = out if isinstance(out, list) else [out]
        top = sorted(out, key=lambda x: -x["score"])[:2]
        for r in top:
            hit = hf_map.get(r["label"])
            n_mapped += hit is not None
            print(
                f"  {s[:38]:<40} -> label={r['label']!r:<12} "
                f"score={r['score']:.3f}  _HF_MAP.get -> {hit}"
            )
    print(
        f"\nlabel yang berhasil dipetakan: {n_mapped}/{len(samples)}"
        f"\n-> jalur HF {'AKTIF' if n_mapped else 'MATI'}; "
        f"{'' if n_mapped else 'semua baris memakai _lex_one (leksikon).'}"
    )
    return 0


def prepare(data_path: Path):
    """
    Tahap pembersihan cell 3, berhenti tepat sebelum pelabelan.

    Mengembalikan frame 19.491 baris — populasi yang dipakai cell 3 saat
    mencetak 'rows after prep' dan 'emoji rate'. Dipisah dari build() supaya
    verify_reproduction.py bisa memeriksa kedua angka itu pada tahap yang benar,
    bukan pada korpus berlabel yang sudah tersaring.
    """
    import pandas as pd

    df = (
        pd.read_excel(data_path)[["full_text"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df["full_text"] = df["full_text"].astype(str).str.slice(0, 300)
    df = df[df["full_text"].str.strip().astype(bool)].reset_index(drop=True)
    df["text_norm"] = df["full_text"].apply(lambda s: apply_slang(_norm(s)))
    df["emoji_count"] = df["full_text"].apply(lambda s: len(EMOJI_RX.findall(s)))
    return df[df["text_norm"].str.strip().astype(bool)].reset_index(drop=True)


def build(data_path: Path, out_path: Path, verify: bool) -> int:
    if not data_path.exists():
        print(f"ERROR: dataset tidak ditemukan: {data_path}", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"baca  : {data_path}")

    df = prepare(data_path)
    emoji_rate = df["emoji_count"].gt(0).mean() * 100
    n_prep = len(df)
    print(f"prep  : {n_prep} baris | emoji rate: {emoji_rate:.1f}%")

    # --- pelabelan ---------------------------------------------------------
    df["labels"] = [lex_one(t) for t in df["text_norm"]]
    df = df[df["labels"].apply(sum) >= 1].reset_index(drop=True)
    n_lab = len(df)
    counts = {c: int(sum(r[i] for r in df["labels"])) for i, c in enumerate(LABEL_COLS)}

    print(f"label : {n_lab} baris ({time.time() - t0:.1f}s)")
    for c, n in counts.items():
        print(f"        {c}: {n}/{n_lab}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["text_norm", "emoji_count", "labels"]].to_excel(out_path, index=False)
    print(f"simpan: {out_path}")

    if not verify:
        return 0

    # --- uji regresi terhadap keluaran notebook ----------------------------
    print("\nkesetaraan dengan keluaran cell 3:")
    checks = [
        ("baris setelah prep", n_prep, NOTEBOOK_ROWS_AFTER_PREP),
        ("baris berlabel", n_lab, NOTEBOOK_ROWS_LABELED),
        *(
            (f"label {c}", counts[c], NOTEBOOK_LABEL_COUNTS[c])
            for c in LABEL_COLS
        ),
    ]
    ok = True
    for name, got, want in checks:
        same = got == want
        ok &= same
        print(f"  {'OK  ' if same else 'BEDA'}  {name:<20} skrip={got:<7} notebook={want}")

    rate_ok = abs(emoji_rate - NOTEBOOK_EMOJI_RATE) < 0.05
    ok &= rate_ok
    print(
        f"  {'OK  ' if rate_ok else 'BEDA'}  {'emoji rate':<20} "
        f"skrip={emoji_rate:.1f}%  notebook={NOTEBOOK_EMOJI_RATE}%"
    )

    print(
        "\n-> korpus identik dengan yang dipakai notebook."
        if ok
        else "\n-> BERBEDA dari notebook. Jangan latih dari korpus ini sebelum"
        " selisihnya dijelaskan."
    )
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="XLSX mentah")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="XLSX berlabel")
    p.add_argument(
        "--no-verify", action="store_true", help="lewati perbandingan dengan notebook"
    )
    p.add_argument(
        "--probe-hf-mapping",
        action="store_true",
        help="jalankan model pelabel & tunjukkan bahwa _HF_MAP tidak cocok",
    )
    a = p.parse_args()

    if a.probe_hf_mapping:
        return probe_hf_mapping()
    return build(a.data, a.out, verify=not a.no_verify)


if __name__ == "__main__":
    raise SystemExit(main())
