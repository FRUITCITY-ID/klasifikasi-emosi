# Checkpoint model

Letakkan checkpoint IndoBERT-base hasil training di folder ini dengan nama:

```text
best_indobert_base.pt
```

SIPEMO hanya mengaktifkan prediksi setelah struktur checkpoint cocok dengan
`indobenchmark/indobert-base-p1`. File model dasar dari Hugging Face tidak cukup,
karena classification head untuk empat label harus sudah dilatih.

Jika checkpoint berada di tempat lain, set `SIPEMO_CHECKPOINT` ke path absolut
sebelum menjalankan aplikasi.

## Membuat ulang

Dari folder proyek, tanpa membuka notebook (±20 menit pada CPU):

```bash
.venv\Scripts\python training/build_corpus.py && .venv\Scripts\python training/train_indobert.py
```

Skrip pertama membangun `data/labeled_silver.xlsx` dari `tugas-akhir_cleaned.xlsx`
dan memverifikasinya terhadap output cell 3; skrip kedua melatih IndoBERT-base dan
menulis `best_indobert_base.pt` ke folder ini.

