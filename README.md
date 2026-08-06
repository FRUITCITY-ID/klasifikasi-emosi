# SIPEMO — Sistem Pemantauan Emosi Publik

Antarmuka penelitian untuk `multilabel_bert_comparison.ipynb`: klasifikasi emosi
multi-label (`senang`, `sedih`, `marah`, `takut`) pada teks media sosial berbahasa
Indonesia.

Backend menjalankan model sungguhan dengan pipeline yang identik dengan notebook —
bukan simulasi leksikon. Kalau checkpoint tidak tersedia atau arsitekturnya tidak
cocok, sistem **menolak memprediksi** dan menjelaskan sebabnya, bukan menampilkan
angka yang menyesatkan.

---

## Struktur

```
Sipemo/
├── backend/
│   ├── app.py              FastAPI — endpoint & penyajian frontend
│   ├── config.py           Konstanta dari notebook (label, MAX_LEN, threshold, path)
│   ├── preprocessing.py    Port 1:1 _norm / SLANG / EMOJI_RX / LEX dari notebook
│   ├── model.py            Muat checkpoint + verifikasi arsitektur + inferensi
│   ├── research.py         Metrik hasil eksekusi notebook (beku, tidak dihitung ulang)
│   ├── test_pipeline.py    Uji paritas preprocessing vs notebook
│   ├── verify_research.py  88 uji: tiap angka di UI vs output notebook
│   ├── probe_neutral.py    Ukur perilaku model pada teks netral & teks kosong
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/                tokens · base · components
│   └── js/                 api · charts · format · views · main
├── training/
│   ├── build_corpus.py         XLSX mentah → data/labeled_silver.xlsx (cell 3)
│   ├── train_indobert.py       korpus → models/best_indobert_base.pt (cell 5-9)
│   ├── verify_reproduction.py  34 pemeriksaan checkpoint lokal vs output notebook
│   └── requirements.txt
├── arsip/                      prototipe lama — TIDAK berlaku, lihat arsip/README.md
├── tugas-akhir_cleaned.xlsx    dataset mentah, 20.559 baris
└── multilabel_bert_comparison.ipynb
```

---

## Menjalankan

```bash
python -m venv .venv && .venv\Scripts\pip install -r backend/requirements.txt
```

```bash
.venv\Scripts\python -m uvicorn app:app --port 8000 --app-dir backend
```

Buka <http://127.0.0.1:8000>.

> **Soal versi `transformers`.** Notebook dijalankan pada transformers 4.44.0 /
> Python 3.10.0, tetapi `requirements.txt` sengaja tidak mem-pin 4.44.0.
> Versi itu menarik `tokenizers` 0.19.x — ekstensi Rust yang terbit sebelum
> Python 3.13, jadi tanpa wheel `cp313`. Di Python 3.13 pip terpaksa
> membangunnya dari sumber, gagal menyiapkan toolchain Rust, dan seluruh
> instalasi berhenti. Yang menggantikan pin itu adalah bukti: lingkungan proyek
> ini (transformers 4.57.6 / Python 3.13.14 / torch 2.13.0+cpu) lolos
> `verify_reproduction.py` **34/34**, termasuk 876 keputusan prediksi test set
> yang identik elemen per elemen dengan notebook. Kalau Anda mengganti versi,
> jalankan ulang skrip itu.

Tab **Evaluasi Model** dan **Dataset & Metode** langsung berfungsi tanpa checkpoint —
isinya angka beku dari notebook. Tab **Analisis** butuh checkpoint.

### Variabel lingkungan

| Variabel | Default | Fungsi |
|---|---|---|
| `SIPEMO_CHECKPOINT` | `models\best_indobert_base.pt` | Lokasi file `.pt`; path relatif dihitung dari folder proyek |
| `SIPEMO_FORCE_CPU` | `0` | Paksa CPU walau ada CUDA |
| `SIPEMO_ALLOW_MISMATCH` | `0` | Izinkan checkpoint yang arsitekturnya tidak cocok — **tidak disarankan** |
| `SIPEMO_PORT` | `8000` | Port server |

---

## Menyiapkan checkpoint

> **Penting.** Jangan gunakan `best_bert.pt` dari eksekusi notebook versi lama;
> file itu **bukan** bobot IndoBERT-base.

Pada versi awal cell 9, `train_one()` menyimpan ke path yang sama untuk setiap model:

```python
torch.save({...}, r'C:\MULTI LABEL\Hasil\best_bert.pt')
```

Karena berada di dalam loop `for name, path in MODELS.items()`, setiap model menimpa
file sebelumnya. Yang tersisa di disk adalah bobot model **yang dilatih terakhir**,
yaitu `mBERT-base` — bukan `IndoBERT-base (baseline)` yang F1-nya tertinggi.

Cell 15 sebenarnya mendeteksi hal ini lewat loop pencocokan `vocab_size`, lalu
diam-diam beralih ke arsitektur yang cocok. Akibatnya header yang tercetak —
`Model : IndoBERT-base (baseline)`, `F1 M : 0.994` — berada di atas prediksi yang
sebenarnya dihasilkan mBERT (F1-macro 0.742). Itulah sebabnya demo di notebook
memberi `sedih 0.536` lebih tinggi daripada `senang 0.424` untuk kalimat
*"hasilnya jauh lebih bagus dari yang saya kira"*.

Backend ini memverifikasi checkpoint sebelum memakainya, jadi kekeliruan itu tidak
akan terulang di antarmuka.

Versi cell 9 yang sekarang sudah memakai `CHECKPOINT_FILES` — tiap model punya
file sendiri, jadi menjalankan ulang notebook juga menghasilkan
`models/best_indobert_base.pt` yang benar. Peringatan di atas hanya berlaku untuk
`best_bert.pt` peninggalan versi lama.

### Cara menghasilkan checkpoint tanpa membuka notebook

Dua skrip di `training/` mengerjakan hal yang sama dari terminal, dan keduanya
mengimpor `backend/preprocessing.py` — bukan menyalin ulang fungsinya — sehingga
teks yang dipakai saat training dijamin identik dengan yang dipakai backend saat
inferensi.

```bash
.venv\Scripts\pip install -r training/requirements.txt
```

```bash
.venv\Scripts\python training/build_corpus.py
```

Membaca `tugas-akhir_cleaned.xlsx`, menjalankan preprocessing + pelabelan cell 3,
lalu menulis `data/labeled_silver.xlsx`. Selesai dalam ±2 detik dan otomatis
membandingkan hasilnya dengan output cell 3 — jumlah baris, sebaran empat label,
dan emoji rate harus cocok semua sebelum korpus dianggap sah.

```bash
.venv\Scripts\python training/train_indobert.py
```

Melatih **IndoBERT-base saja** dengan split, hyperparameter, sweep threshold, dan
isi checkpoint yang sama seperti `train_one()` cell 9, lalu menyimpan
`models/best_indobert_base.pt`. Sekitar 20 menit pada CPU 6-thread; tambahkan
`--benchmark 6` untuk mengukur dulu tanpa melatih.

Jika checkpoint disimpan di lokasi lain, arahkan backend secara eksplisit:

```bash
set SIPEMO_CHECKPOINT=D:\lokasi-model\best_indobert_base.pt
```

### Membuktikan checkpoint lokal setara dengan hasil di skripsi

```bash
.venv\Scripts\python training/verify_reproduction.py
```

34 pemeriksaan, dan **tidak satu pun angkanya diketik ulang oleh manusia** —
semua pembanding di-parse langsung dari output tersimpan di notebook (cell 3, 4,
5, 9), lalu dibandingkan dengan hasil yang dihitung ulang dari
`data/labeled_silver.xlsx` dan `models/best_indobert_base.pt`.

| Lapis | Yang diperiksa | Sumber |
|---|---|---|
| 1 | baris prep, emoji rate, baris berlabel, 4 jumlah label | cell 3 |
| 2 | kardinalitas presisi penuh, 5 kombo teratas, panjang p50/p95 | cell 4 |
| 3 | ukuran train/val/test | cell 5 |
| 4 | **`y_true_test` 219×4, elemen per elemen** | cell 9 |
| 5 | **`y_pred_test` 219×4, elemen per elemen** | cell 9 |
| 6 | F1 micro/macro/samples, Hamming, SubsetAcc, F1 per label — presisi penuh | cell 9 |
| 7 | loss, val F1-macro, threshold tiap epoch | cell 9 |

Lapis 4 dan 5 yang paling menentukan. Lapis 4 mencocokkan 876 nilai ground-truth:
kalau semuanya sama, korpus, urutan baris, dan split-nya identik — bukan sekadar
totalnya kebetulan sama. Lapis 5 mencocokkan 876 keputusan biner model: bobot
lokal menghasilkan prediksi yang persis sama dengan yang dilaporkan di skripsi,
termasuk 3 baris yang sama-sama salah prediksi.

Batas yang jujur: lapis 7 hanya bisa dibandingkan sampai 4 desimal, karena cell 9
mencetak `loss={:.4f}`. Dan skrip ini membuktikan **kesetaraan perilaku**, bukan
bahwa bobotnya identik bit-per-bit — file `.pt` dari eksekusi notebook asli tidak
tersedia untuk dibandingkan langsung.

---

## Verifikasi checkpoint

Sebelum memuat bobot, `model.py` menjalankan dua pemeriksaan.

**Gate 1 — lebar embedding (mengikat).** Lebar embedding checkpoint harus sama
dengan model yang dimuat. Kalau tidak, model tidak dimuat dan API melaporkan status
`mismatch`. Ini yang menangkap kasus checkpoint mBERT (119.547) dan
XLM-RoBERTa (250.002).

**Gate 2 — sidik jari F1 (peringatan).** Notebook menyimpan val F1-macro terbaik ke
key `f1_macro`. Nilainya berbeda-beda per model, jadi bisa dipakai memastikan
checkpoint milik model yang benar:

| Model | val F1-macro | lebar embedding | `len(tokenizer)` |
|---|---|---|---|
| IndoBERT-IndoNLU | 0.9880 | 50.000 | 30.521 |
| **IndoBERT-base (baseline)** | **0.9789** | **50.000** | **30.521** |
| IndoBERTweet | 0.8658 | 31.923 | 31.923 |
| mBERT-base | 0.7816 | 119.547 | 119.547 |
| IndoBERT-IndoLEM | 0.7582 | 31.923 | 31.923 |
| XLM-RoBERTa-base | 0.6746 | 250.002 | 250.002 |

Gate 1 tidak bisa membedakan p1 dari p2, juga IndoBERTweet dari IndoLEM, karena
lebarnya sama — di situlah gate 2 berguna. Simpan tiap model ke nama file berbeda
supaya tidak tertukar.

### Kenapa backend tidak memanggil `resize_token_embeddings()`

Perhatikan dua kolom terakhir tabel di atas. Untuk IndoBERT-base, `config.vocab_size`
adalah **50.000** sedangkan tokenizer-nya hanya punya **30.521** token — ketidakcocokan
bawaan rilis IndoBERT.

Cell 9 (yang menghasilkan bobot) melatih **tanpa** resize, jadi checkpoint IndoBERT-base
yang benar lebarnya 50.000. Versi awal cell 15 memanggil `mdl_i.resize_token_embeddings(len(tok_i))`
yang akan menyusutkannya ke 30.521 — sehingga `load_state_dict` pasti gagal dengan
*size mismatch* untuk checkpoint IndoBERT-base yang sah.

Versi awal cell 15 tidak pernah menampakkan kegagalan itu karena checkpoint di disk adalah
milik mBERT, satu-satunya kasus di mana kedua angka kebetulan sama (119.547). Backend
sekarang mengikuti cell 9, bukan cell 15: bentuk bobot ditentukan oleh training.

---

## Kesetaraan dengan notebook

`backend/test_pipeline.py` menyalin ulang fungsi asli dari notebook secara
independen lalu membandingkannya dengan implementasi backend:

```bash
python backend/test_pipeline.py
```

Yang diuji pada 16 kasus: `_norm()`, substitusi `SLANG`, `EMOJI_RX.findall()`,
`_label_emoji_set()`, dan kesamaan isi kamus `SLANG` / `LEX` / `LABEL_COLS`.

`backend/verify_backend.py` menguji lapisan model: penolakan checkpoint arsitektur
lain, penerimaan checkpoint yang sah, peringatan sidik jari F1, struktur keluaran
`predict()`, dan kebenaran fusi OR. Skrip ini memakai bobot acak — yang diverifikasi
mekanika pipeline, bukan kualitas prediksi.

```bash
.venv\Scripts\python backend/verify_backend.py
```

### Angka yang tampil di UI

```bash
.venv\Scripts\python backend/verify_research.py
```

88 pemeriksaan. `backend/research.py` adalah satu-satunya sumber angka untuk tab
Evaluasi Model dan Dataset & Metode, dan isinya diketik manusia — jadi bisa salah
salin. Skrip ini membandingkan setiap nilainya dengan output tersimpan di
notebook: 6 model × (5 metrik agregat + 4 F1 per label + threshold + 3 epoch loss
+ 3 epoch val F1), seluruh statistik dataset, hyperparameter, versi runtime, dan
pemilihan model terbaik. Tanpa dependensi di luar pustaka standar.

Skrip ini juga memeriksa frontend: `frontend/js/` tidak boleh memuat angka
penelitian maupun angka turunan (persentase dihitung dari nilai API, bukan
diketik), dan tidak boleh memuat klaim pelabelan yang sudah terbukti keliru
(`skor ≥ 0.30`, `top-k=2`).

### Batas sistem yang ikut diukur

```bash
.venv\Scripts\python backend/probe_neutral.py
```

Menjalankan 15 kalimat netral dan 5 teks yang habis setelah `_norm()` melalui
model, lalu membandingkan hasilnya dengan peringatan yang tertulis di antarmuka.
Keluar dengan status ≠ 0 kalau keduanya tidak lagi cocok — misalnya setelah
melatih ulang dengan kelas netral, yang berarti caveat di `research.py` perlu
diperbarui. Lihat poin 5 pada "Catatan untuk sidang".

Satu catatan presisi: threshold dibandingkan pada 2 desimal, sengaja. Cell 8
menyapu `np.arange(0.20, 0.55, 0.05)` yang menghasilkan `0.39999999999999997` dan
`0.44999999999999996` alih-alih 0.40 dan 0.45. Tabel cell 11 — rujukan yang masuk
skripsi — mencetak 0.40 dan 0.45, dan `research.py` mengikuti tabel itu. Menuntut
presisi penuh berarti menuntut research.py menyalin artefak float, bukan angka
penelitian.

Alur inferensi mengikuti `predict_one()` cell 15 persis:

```
norm      = SLANG(_norm(raw))
prob      = sigmoid(model(tokenizer(norm)).logits)
bin_bert  = prob >= threshold
bin_emoji = leksikon_emoji(EMOJI_RX.findall(raw))
bin_fused = bin_bert | bin_emoji
```

### Fusi terjadi di level keputusan

Emoji **tidak** masuk ke encoder. Regex `[^a-z0-9\s]` pada `_norm()` menghapus semua
emoji sebelum teks mencapai tokenizer. Emoji dicocokkan terpisah ke leksikon, lalu
digabung dengan OR bitwise setelah keduanya menghasilkan keputusan biner.

Tidak ada Emoji2Vec, tidak ada leksikon emoticon, dan tidak ada konkatenasi vektor
fitur dalam penelitian ini — prototipe lama (`arsip/sipemo-lama.html`) menampilkan ketiganya,
padahal tidak ada di notebook.

### Kekhasan yang sengaja dipertahankan

`EMOJI_RX` memuat `❤️` di dalam character class. Di Python itu dua codepoint
terpisah (`U+2764` dan `U+FE0F`), sehingga pola mencocokkan masing-masing sendiri,
sementara `LEX['senang']` berisi `'❤️'` sebagai satu string dua-karakter — jadi ❤️
tidak pernah benar-benar cocok lewat jalur emoji. Perilaku ini dibiarkan apa adanya
agar backend identik dengan notebook. Kalau ingin diperbaiki, ubah di notebook dulu
lalu latih ulang.

---

## Catatan untuk sidang

Angka-angka pada tab Evaluasi tinggi (F1-macro 0.994). Lima hal berikut sebaiknya
Anda sampaikan lebih dulu sebelum ditanyakan penguji — semuanya sudah ditampilkan
apa adanya di tab **Dataset & Metode**:

1. **Pelabelnya leksikon kata kunci, bukan model RoBERTa.** Cell 3 memuat
   `StevenLimcorn/indonesian-roberta-base-emotion-classifier` lalu menjalankannya
   pada 19.491 baris selama 3.246 detik — tetapi keluarannya tidak pernah terpakai:

   ```python
   _HF_MAP = {'Senang':0,'Sedih':1,'Marah':2,'Takut':3}   # kunci di notebook
   # id2label model itu: sadness · anger · love · fear · happy
   ```

   Karena tidak ada kunci yang cocok, `_HF_MAP.get(...)` selalu `None`, `bins`
   tetap `[0,0,0,0]`, syarat `sum(b) > 0` selalu gagal, dan **setiap baris jatuh
   ke `_lex_one()`**. Baris `mode=HF top-k=2` tetap tercetak karena penanda itu
   hanya mengecek apakah pipeline berhasil dimuat, bukan apakah keluarannya
   dipakai.

   Dibuktikan lewat reproduksi: menjalankan jalur leksikon **saja** menghasilkan
   tujuh angka yang identik dengan output cell 3 — 19.491 → 2.188 baris,
   1492/297/170/333 per label, emoji rate 26,3%. Kalau jalur RoBERTa pernah
   menyumbang satu label saja, angka-angka itu mustahil cocok. Jalankan
   `python training/build_corpus.py` untuk mengulang pembuktiannya, atau
   `python training/build_corpus.py --probe-hf-mapping` untuk melihat langsung
   label apa yang sebenarnya dikeluarkan model tersebut.

2. **Karena itu F1 0.994 nyaris tautologis.** Label adalah fungsi deterministik
   dari kata-kata pada `text_norm`, dan `text_norm` itu juga yang dibaca IndoBERT.
   Jadi angka tersebut menunjukkan IndoBERT sanggup menghafal aturan kata kunci —
   bukan bukti akurasi terhadap emosi sebenarnya. Ini juga menjelaskan kenapa
   hanya 2.188 dari 19.491 baris (11%) yang terpakai: sisanya tidak memuat satu
   pun kata kunci. Pemicu terbanyak per label: senang ← `love` (418), `good` (250);
   sedih ← `sedih` (126), `galau` (86); marah ← `emosi` (52), `marah` (43);
   takut ← `anxious` (136), `takut` (85).

3. **Kelas marah sangat kecil.** Hanya 170 dari 2.188 baris. XLM-RoBERTa mendapat
   F1 0.00 pada label ini dan IndoBERTweet 0.30 — F1-macro mereka jatuh terutama
   karena kelas ini.

4. **Datanya hampir single-label.** Rata-rata 1.05 label per tweet; 1.429 dari 2.188
   baris hanya berlabel senang. Kemampuan multi-label jarang teruji pada data ini.

5. **Tidak ada kelas netral — dan ini yang paling mungkin muncul saat demo.**
   Baris tanpa kata kunci dibuang saat pelabelan, jadi model tidak pernah melihat
   contoh "tidak ada emosi" dan tidak punya tempat untuk menaruhnya. Ditambah
   1.492 dari 2.188 baris latih berlabel senang dan ambang hasil tuning hanya
   0,30, kalimat datar cenderung keluar sebagai **senang**:

   ```bash
   .venv\Scripts\python backend/probe_neutral.py
   ```

   Pengukuran terakhir: **14 dari 15** kalimat netral dilabeli senang, dan
   **5 dari 5** teks yang habis setelah `_norm()` (`"..."`, `"???"`) juga
   menghasilkan senang — pada masukan kosong tokenizer hanya menerima
   `[CLS] [SEP]` sehingga model mengembalikan prior-nya.

   Antarmuka sudah mengatakannya di tiga tempat: banner tetap di atas kotak
   input, tombol contoh **"Batas: kalimat netral"** supaya Anda bisa
   mendemokannya sendiri, dan peringatan khusus yang muncul kalau teks habis
   setelah normalisasi. Skrip di atas keluar dengan status ≠ 0 kalau hasil
   pengukurannya tidak lagi cocok dengan teks caveat — jadi peringatan itu tidak
   bisa diam-diam jadi basi.

   Ini bukan cacat implementasi, melainkan konsekuensi desain korpus, dan lebih
   baik Anda yang menunjukkannya lebih dulu.

Catatan teknis untuk poin 1: entri emoji di dalam `_LEX` (`😊`, `😭`, …) tidak
pernah aktif saat pelabelan, karena label dihitung atas `text_norm` sedangkan
`_norm()` sudah menghapus semua emoji lewat `[^a-z0-9\s]`. Diperiksa ulang atas
2.188 baris: nol kecocokan. Emoji baru berperan saat inferensi, lewat fusi OR
di cell 15.

Satu koreksi terhadap prototipe lama: file itu menampilkan **IndoBERTweet** sebagai
model terbaik. Menurut hasil notebook, IndoBERTweet justru urutan kelima dari enam
(F1-macro 0.725); yang terbaik adalah **IndoBERT-base (baseline)** (0.994). File
tersebut sudah dipindah ke `arsip/sipemo-lama.html` supaya tidak lagi tampak sebagai
bagian sistem — daftar lengkap klaimnya yang keliru ada di `arsip/README.md`.

---

## Endpoint

| Method | Path | Keterangan |
|---|---|---|
| `GET` | `/api/health` | Status model, threshold, device, peringatan |
| `POST` | `/api/predict` | `{"text": "..."}` → probabilitas + jejak pipeline lengkap |
| `POST` | `/api/predict/batch` | `{"texts": [...]}` maksimal 50 teks |
| `GET` | `/api/research` | Metrik 6 model, statistik dataset, hyperparameter, caveat |
| `GET` | `/api/lexicon` | Isi kamus SLANG & leksikon — mengisi kartu "Isi leksikon pelabel" |
| `POST` | `/api/reload` | Muat ulang model setelah checkpoint diganti |

---

## Catatan antarmuka

Palet emosi diambil dari palet data-viz tervalidasi dan lolos enam pemeriksaan
(lightness band, chroma floor, separasi colorblind, normal-vision floor, kontras)
pada mode terang maupun gelap:

| Emosi | Terang | Gelap |
|---|---|---|
| senang | `#eda100` | `#c98500` |
| sedih | `#2a78d6` | `#3987e5` |
| marah | `#e34948` | `#e66767` |
| takut | `#4a3aa7` | `#9085e9` |

Warna tidak pernah menjadi satu-satunya pembawa makna — setiap grafik punya label
langsung dan tombol alih ke tampilan tabel. Mode gelap disusun ulang untuk permukaan
gelap, bukan sekadar pembalikan warna.

**Teks kecil.** `--text-muted` memakai nilai berbeda per tema (`#6b6a65` terang,
`#9c9a93` gelap). Sebelumnya keduanya `#898781`, sehingga teks 11px di mode terang
turun ke 3,0–3,5:1 — di bawah ambang WCAG AA 4,5:1 untuk teks kecil. Nilai sekarang
diukur pada permukaan tergelap tiap tema, yaitu kasus terburuknya: 4,53:1 dan 4,97:1.

**Angka.** Seluruh angka yang tampil sebagai data melewati `frontend/js/format.js`
dengan locale `id-ID`: ribuan pakai titik, desimal pakai koma. Sebelumnya halaman
yang sama menampilkan `2.188 baris` (titik = ribuan) berdampingan dengan `88.8%`
(titik = desimal) — karakter sama, dua arti. Satu pengecualian yang disengaja: blok
`.formula` menampilkan ekspresi Python apa adanya, jadi di sana `0.30` tetap `0.30`
karena itu literal kode.
