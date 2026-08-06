"""
Uji paritas: preprocessing backend vs notebook penelitian.

File ini menyalin ulang fungsi asli dari `multilabel_bert_comparison.ipynb`
(cell 3 & 15) secara independen, lalu membandingkannya dengan implementasi di
preprocessing.py. Tujuannya membuktikan backend memproses teks persis sama
dengan yang dipakai saat training — bukan sekadar "mirip".

Jalankan dari folder backend/:
    python test_pipeline.py
"""

from __future__ import annotations

import re
import sys

import preprocessing as pp

# ==========================================================================
# SALINAN LANGSUNG DARI NOTEBOOK — jangan dirapikan, biarkan apa adanya.
# ==========================================================================
SLANG = {'gk':'tidak','ga':'tidak','gak':'tidak','nggak':'tidak','kagak':'tidak','tdk':'tidak',
 'bgt':'banget','emg':'memang','emang':'memang','udh':'sudah','udah':'sudah','skrg':'sekarang',
 'trs':'terus','dr':'dari','klo':'kalau','kalo':'kalau','kl':'kalau','jg':'juga','aja':'saja',
 'doang':'saja','sm':'sama','sy':'saya','aq':'aku','gw':'saya','gue':'saya','lu':'kamu',
 'loe':'kamu','kmu':'kamu','km':'kamu','org':'orang','temen':'teman','tmn':'teman',
 'dmn':'dimana','gimana':'bagaimana','gmn':'bagaimana','krn':'karena','karna':'karena',
 'bbrp':'beberapa','byk':'banyak','dikit':'sedikit','dkt':'sedikit'}
EMOJI_RX = re.compile(r'[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F900-\U0001F9FF❤️]')
URL_RX   = re.compile(r'https?://\S+|www\.\S+'); MENT_RX = re.compile(r'@\w+')
HASH_RX  = re.compile(r'#(\w+)'); WS_RX = re.compile(r'\s+'); REPEAT_RX = re.compile(r'(.)\1{2,}')

LABEL_COLS = ['senang','sedih','marah','takut']

_LEX = {
 'senang':{'senang','bahagia','gembira','suka','rindu','syukur','alhamdulillah','semangat','antusias','bangga','puas','lega','hebat','keren','mantap','good','love','happy','grateful','amazing','nice','seneng','😊','😄','😍','🥰','❤️','💕','👍','🎉'},
 'sedih' :{'sedih','pilu','nelangsa','muram','galau','hampa','kecewa','menyesal','kehilangan','menangis','😭','😢','😔','😞','💔'},
 'marah' :{'marah','kesal','jengkel','sebel','benci','muak','geram','emosi','dongkol','menyebalkan','sialan','bangsat','anjir','bodoamat','😡','😠','🤬'},
 'takut' :{'takut','cemas','khawatir','gelisah','panik','ngeri','horor','paranoid','worried','anxious','afraid','scared','😱','😨','😰'},
}


def _norm(text):
    t = URL_RX.sub(' ', text); t = MENT_RX.sub(' ', t); t = HASH_RX.sub(r' \1 ', t)
    t = t.lower(); t = REPEAT_RX.sub(r'\1\1', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return WS_RX.sub(' ', t).strip()


def notebook_text_norm(raw):
    """Baris yang dipakai cell 3 & cell 15 untuk membuat text_norm."""
    return ' '.join(SLANG.get(w, w) for w in _norm(raw).split())


def _label_emoji_set(em):
    found = [0]*len(LABEL_COLS)
    for e in em:
        for i, c in enumerate(LABEL_COLS):
            if e in _LEX.get(c, set()): found[i] = 1
    return found


# ==========================================================================
CASES = [
    "Akhirnya wisuda juga! Bahagia banget 🥳🎉 perjuangan selama ini terbayar :D",
    "Sedih banget kehilangan sahabat sejak kecil 😭💔 rasanya hampa",
    "Kesel bgt sm pelayanan yg lambat ini 😡 bikin naik darah!",
    "Deg-degan mau sidang besok, takut gak bisa jawab penguji 😨",
    "Seneng lulus tp sedih hrs pindah kota 😊😢 campur aduk",
    "@budi cek https://contoh.co.id/artikel ya #SkripsiLife gw udh capek bgt",
    "AAAAA senengggg bangetttt!!! 🎉🎉🎉",
    "Wah, ternyata hasilnya jauh lebih bagus dari yang saya kira.😲✨ :)",
    "gue kagak tau klo dia dmn skrg, tmn2 jg bingung",
    "###   @@@   https://x.com   ",
    "12345 !!! ??? ...",
    "Sayang ❤️ kamu",
    "",
    "   ",
    "Ga ada emoji di sini sama sekali",
    "MARAH BESAR!!! 🤬🤬 muak sm semua ini",
]

failures: list[str] = []


def check(name: str, got, want, ctx: str = "") -> None:
    if got != want:
        failures.append(f"{name}\n    input : {ctx!r}\n    got   : {got!r}\n    want  : {want!r}")


print("Uji paritas preprocessing: backend vs notebook\n" + "=" * 60)

for text in CASES:
    check("_norm()", pp._norm(text), _norm(text), text)
    check("normalize()", pp.normalize(text), notebook_text_norm(text), text)

    final, steps, _changes = pp.normalize_traced(text)
    check("normalize_traced() == normalize()", final, pp.normalize(text), text)

    ems = pp.find_emojis(text)
    check("find_emojis()", ems, EMOJI_RX.findall(text), text)
    check("label_emoji_set()", pp.label_emoji_set(ems), _label_emoji_set(ems), text)

# Kamus harus identik isinya.
check("kamus SLANG", pp.SLANG, SLANG)
check("leksikon LEX", pp.LEX, _LEX)
check("LABEL_COLS", pp.LABEL_COLS, LABEL_COLS)

n = len(CASES)
if failures:
    print(f"\nGAGAL — {len(failures)} ketidakcocokan pada {n} kasus uji:\n")
    for f in failures:
        print("  ✗ " + f)
    sys.exit(1)

print(f"\n  OK  {n} kasus uji, semua tahap cocok persis dengan notebook.")
print("      _norm · SLANG · EMOJI_RX · LEX · label_emoji_set")

# Tampilkan satu contoh jejak supaya mudah diperiksa manual.
demo = CASES[5]
final, steps, changes = pp.normalize_traced(demo)
print(f"\nContoh jejak — {demo!r}")
for s in steps:
    mark = "*" if s["changed"] else " "
    print(f"  {mark} {s['label']:<26} {s['text']!r}")
print(f"  slang diubah: {changes}")
