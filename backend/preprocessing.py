"""
SIPEMO — preprocessing.

Port 1:1 dari notebook penelitian `multilabel_bert_comparison.ipynb`:
  * SLANG, EMOJI_RX, URL_RX, MENT_RX, HASH_RX, WS_RX, REPEAT_RX  -> cell 3
  * _norm()                                                      -> cell 3 & 15
  * _LEX, _label_emoji_set()                                     -> cell 3 & 15

`normalize()` di bawah menghasilkan string yang identik dengan
`' '.join(SLANG.get(w, w) for w in _norm(raw).split())` pada notebook —
itulah teks persis yang masuk ke tokenizer saat training maupun inferensi.

`normalize_traced()` menjalankan tahapan yang sama tetapi merekam hasil antara
supaya UI bisa menampilkan jejak pipeline. Hasil akhirnya dijamin sama dengan
`normalize()` (lihat tests/test_parity.py).
"""

from __future__ import annotations

import re
from typing import TypedDict

from config import LABEL_COLS

# --------------------------------------------------------------------------
# Kamus slang (notebook cell 3) — disalin apa adanya, urutan dipertahankan.
# --------------------------------------------------------------------------
SLANG: dict[str, str] = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "kagak": "tidak", "tdk": "tidak",
    "bgt": "banget", "emg": "memang", "emang": "memang", "udh": "sudah",
    "udah": "sudah", "skrg": "sekarang",
    "trs": "terus", "dr": "dari", "klo": "kalau", "kalo": "kalau",
    "kl": "kalau", "jg": "juga", "aja": "saja",
    "doang": "saja", "sm": "sama", "sy": "saya", "aq": "aku", "gw": "saya",
    "gue": "saya", "lu": "kamu",
    "loe": "kamu", "kmu": "kamu", "km": "kamu", "org": "orang",
    "temen": "teman", "tmn": "teman",
    "dmn": "dimana", "gimana": "bagaimana", "gmn": "bagaimana",
    "krn": "karena", "karna": "karena",
    "bbrp": "beberapa", "byk": "banyak", "dikit": "sedikit", "dkt": "sedikit",
}

# --------------------------------------------------------------------------
# Regex (notebook cell 3) — disalin apa adanya.
# Catatan: '❤️' di dalam character class adalah DUA codepoint terpisah
# (U+2764 dan U+FE0F), sehingga pola ini mencocokkan masing-masing sendiri.
# Perilaku ini dipertahankan agar identik dengan notebook.
# --------------------------------------------------------------------------
EMOJI_RX = re.compile(
    r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F900-\U0001F9FF❤️]"
)
URL_RX = re.compile(r"https?://\S+|www\.\S+")
MENT_RX = re.compile(r"@\w+")
HASH_RX = re.compile(r"#(\w+)")
WS_RX = re.compile(r"\s+")
REPEAT_RX = re.compile(r"(.)\1{2,}")

# --------------------------------------------------------------------------
# Leksikon emoji/kata per label (notebook cell 3).
# Saat inferensi hanya entri emoji yang bisa cocok, karena _label_emoji_set()
# hanya diberi hasil EMOJI_RX.findall() dari teks mentah.
# --------------------------------------------------------------------------
LEX: dict[str, set[str]] = {
    "senang": {
        "senang", "bahagia", "gembira", "suka", "rindu", "syukur",
        "alhamdulillah", "semangat", "antusias", "bangga", "puas", "lega",
        "hebat", "keren", "mantap", "good", "love", "happy", "grateful",
        "amazing", "nice", "seneng", "😊", "😄", "😍", "🥰", "❤️", "💕", "👍", "🎉",
    },
    "sedih": {
        "sedih", "pilu", "nelangsa", "muram", "galau", "hampa", "kecewa",
        "menyesal", "kehilangan", "menangis", "😭", "😢", "😔", "😞", "💔",
    },
    "marah": {
        "marah", "kesal", "jengkel", "sebel", "benci", "muak", "geram",
        "emosi", "dongkol", "menyebalkan", "sialan", "bangsat", "anjir",
        "bodoamat", "😡", "😠", "🤬",
    },
    "takut": {
        "takut", "cemas", "khawatir", "gelisah", "panik", "ngeri", "horor",
        "paranoid", "worried", "anxious", "afraid", "scared", "😱", "😨", "😰",
    },
}


class NormStep(TypedDict):
    """Satu tahap pipeline pembersihan teks."""

    key: str
    label: str
    detail: str
    text: str
    changed: bool


def _norm(text: str) -> str:
    """Salinan persis `_norm()` dari notebook cell 3 / cell 15."""
    t = URL_RX.sub(" ", text)
    t = MENT_RX.sub(" ", t)
    t = HASH_RX.sub(r" \1 ", t)
    t = t.lower()
    t = REPEAT_RX.sub(r"\1\1", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return WS_RX.sub(" ", t).strip()


def apply_slang(text: str) -> str:
    """Tahap kedua: substitusi slang per token (notebook cell 3 & 15)."""
    return " ".join(SLANG.get(w, w) for w in text.split())


def normalize(raw: str) -> str:
    """Teks final yang masuk ke tokenizer — identik dengan notebook."""
    return apply_slang(_norm(raw))


def normalize_traced(raw: str) -> tuple[str, list[NormStep], list[dict[str, str]]]:
    """
    Jalankan pipeline yang sama sambil merekam hasil tiap tahap.

    Returns:
        (teks_final, langkah_langkah, perubahan_slang)
    """
    steps: list[NormStep] = []

    def record(key: str, label: str, detail: str, before: str, after: str) -> str:
        steps.append(
            {
                "key": key,
                "label": label,
                "detail": detail,
                "text": after,
                "changed": before != after,
            }
        )
        return after

    t = raw
    t = record("url", "Hapus URL", r"https?://\S+|www\.\S+", t, URL_RX.sub(" ", t))
    t = record("mention", "Hapus mention", r"@\w+", t, MENT_RX.sub(" ", t))
    t = record("hashtag", "Buka hashtag", r"#(\w+) → \1", t, HASH_RX.sub(r" \1 ", t))
    t = record("lower", "Case folding", "text.lower()", t, t.lower())
    t = record(
        "repeat", "Redam huruf berulang", r"(.)\1{2,} → \1\1", t, REPEAT_RX.sub(r"\1\1", t)
    )
    t = record(
        "punct",
        "Buang non-alfanumerik",
        r"[^a-z0-9\s] → spasi (emoji & emoticon ikut terhapus di sini)",
        t,
        re.sub(r"[^a-z0-9\s]", " ", t),
    )
    t = record("space", "Rapikan spasi", r"\s+ → spasi, lalu strip", t, WS_RX.sub(" ", t).strip())

    before_slang = t
    final = apply_slang(t)
    steps.append(
        {
            "key": "slang",
            "label": "Normalisasi slang",
            "detail": f"kamus SLANG ({len(SLANG)} entri)",
            "text": final,
            "changed": before_slang != final,
        }
    )

    changes = [
        {"from": w, "to": SLANG[w]} for w in before_slang.split() if w in SLANG
    ]
    return final, steps, changes


def find_emojis(raw: str) -> list[str]:
    """EMOJI_RX.findall() atas teks MENTAH (notebook cell 15)."""
    return EMOJI_RX.findall(raw)


def label_emoji_set(emojis: list[str]) -> list[int]:
    """Salinan persis `_label_emoji_set()` dari notebook cell 15."""
    found = [0] * len(LABEL_COLS)
    for e in emojis:
        for i, c in enumerate(LABEL_COLS):
            if e in LEX.get(c, set()):
                found[i] = 1
    return found


def emoji_matches(emojis: list[str]) -> list[dict[str, str]]:
    """Rincian emoji mana memicu label mana — untuk ditampilkan di UI."""
    out: list[dict[str, str]] = []
    for e in emojis:
        hits = [c for c in LABEL_COLS if e in LEX.get(c, set())]
        out.append({"emoji": e, "labels": hits})
    return out
