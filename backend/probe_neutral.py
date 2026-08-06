"""
SIPEMO — ukur perilaku model pada teks netral & teks kosong.

Kenapa skrip ini ada
--------------------
Tab Evaluasi menampilkan F1-macro 0.994, dan itu angka yang jujur untuk test
set-nya. Tetapi test set itu seluruhnya berisi baris yang MEMUAT kata kunci
leksikon — baris tanpa kata kunci dibuang saat pelabelan (19.491 -> 2.188).
Akibatnya model tidak pernah melihat satu pun contoh "tidak ada emosi", dan
tidak ada kelas netral untuk ditempati.

Konsekuensinya baru terlihat saat sistem diberi kalimat netral, yang justru
paling mungkin diketik penguji saat demo. Skrip ini mengukurnya, sehingga
peringatan di antarmuka punya angka yang bisa diulang — bukan kesan.

Menjalankan:
    .venv\\Scripts\\python backend/probe_neutral.py

Keluar dengan status != 0 kalau hasil pengukuran TIDAK lagi cocok dengan
peringatan yang tertulis di antarmuka. Itu bukan kegagalan sistem; itu tanda
bahwa teks caveat di backend/research.py perlu diperbarui (misalnya setelah
melatih ulang dengan kelas netral).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
import model as model_mod  # noqa: E402

# Kalimat netral: pernyataan faktual/administratif tanpa satu pun kata emosi.
# Sengaja dipilih dari ragam yang wajar diketik penguji saat mencoba sistem.
NEUTRAL = [
    "Rapat dimulai pukul sembilan pagi di ruang A",
    "Harga beras naik lagi bulan ini",
    "Pemerintah mengumumkan kebijakan baru soal pajak kendaraan",
    "Jadwal kereta api dari Bandung ke Jakarta berubah mulai Senin",
    "Saya membeli dua kilogram gula di pasar",
    "Cuaca hari ini berawan dengan suhu 28 derajat",
    "Skripsi saya membahas klasifikasi emosi multi-label",
    "Nomor antrian saya 42",
    "Populasi Indonesia sekitar 280 juta jiwa",
    "Tolong kirim dokumen itu sebelum jam lima sore",
    "Dia berangkat ke kantor naik motor",
    "Buku itu terdiri dari 300 halaman",
    "Kucing saya tidur di sofa",
    "Data dikumpulkan lewat API Twitter",
    "Besok ada kuliah pengganti di gedung B",
]

# Teks yang habis setelah _norm(): seluruh karakternya dibuang regex
# [^a-z0-9\s], sehingga tokenizer hanya menerima [CLS] [SEP].
DEGENERATE = ["...", "???", "!!!", "---", "😊😊"]


def main() -> int:
    state = model_mod.load()
    if not state.loaded:
        print(f"model tidak siap ({state.status}): {state.message}", file=sys.stderr)
        return 2

    print(f"model     : {config.SERVED_MODEL_NAME}")
    print(f"threshold : {state.threshold:.2f}\n")

    print("A. Kalimat netral")
    print(f"  {'kalimat':<48}{'senang':>8}{'sedih':>8}{'marah':>8}{'takut':>8}  label")
    tally: Counter[str] = Counter()
    n_senang = 0
    for t in NEUTRAL:
        r = model_mod.predict(t)
        p = r["probs"]
        cols = "".join(f"{p[c]:>8.3f}" for c in config.LABEL_COLS)
        print(f"  {t[:46]:<48}{cols}  {r['active'] or ['(tidak ada)']}")
        for a in r["active"] or ["(tidak ada)"]:
            tally[a] += 1
        if "senang" in r["active"]:
            n_senang += 1

    print(f"\n  sebaran label pada {len(NEUTRAL)} kalimat netral: {dict(tally)}")
    print(f"  mengandung 'senang': {n_senang}/{len(NEUTRAL)}")

    print("\nB. Teks yang habis setelah normalisasi")
    # Yang diukur di sini khusus jalur BERT. Jalur emoji sengaja dikecualikan:
    # untuk '😊😊' label senang memang PANTAS muncul lewat fusi leksikon, dan
    # itu perilaku yang benar. Yang jadi soal adalah ketika teks tidak
    # menyisakan apa pun untuk dibaca namun BERT tetap mengeluarkan label.
    degen_senang = 0
    for t in DEGENERATE:
        r = model_mod.predict(t)
        bert_only = [c for c in config.LABEL_COLS if r["pred_bert"][c]]
        if r["pred_bert"]["senang"]:
            degen_senang += 1
        print(f"  {t!r:<12} norm={r['norm']!r:<6} bert={bert_only}  fusi={r['active']}")

    # ---- apakah peringatan di UI masih akurat? ---------------------------
    print("\n" + "=" * 74)
    ok = True

    if n_senang < len(NEUTRAL) * 0.6:
        ok = False
        print(
            f"  BERUBAH: hanya {n_senang}/{len(NEUTRAL)} kalimat netral yang dilabeli\n"
            "           'senang'. Caveat 'Teks netral cenderung jatuh ke senang' di\n"
            "           backend/research.py perlu ditinjau ulang."
        )
    else:
        print(f"  OK  {n_senang}/{len(NEUTRAL)} kalimat netral dilabeli 'senang'")

    if degen_senang < len(DEGENERATE):
        ok = False
        print(
            f"  BERUBAH: hanya {degen_senang}/{len(DEGENERATE)} teks kosong yang jatuh\n"
            "           ke 'senang'. Peringatan teks-kosong di frontend perlu ditinjau."
        )
    else:
        print(f"  OK  {degen_senang}/{len(DEGENERATE)} teks kosong jatuh ke 'senang'")

    print("=" * 74)
    print(
        "Perilaku ini bukan bug kode. Penyebabnya struktural: tidak ada kelas\n"
        "netral, 1.492 dari 2.188 baris latih berlabel senang, dan threshold\n"
        "hasil tuning hanya 0,30."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
