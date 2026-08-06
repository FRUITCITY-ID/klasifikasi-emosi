# Arsip — jangan dipakai sebagai rujukan

Isi folder ini **sudah tidak berlaku**. Disimpan hanya sebagai jejak revisi,
bukan sebagai bagian dari sistem.

## `sipemo-lama.html`

Prototipe antarmuka versi pertama. Berdiri sendiri (tidak memanggil backend) dan
memuat empat klaim yang **bertentangan dengan hasil notebook**:

| Klaim di file lama | Kenyataan menurut `multilabel_bert_comparison.ipynb` |
|---|---|
| Model terbaik = **IndoBERTweet** (ada di `<title>`) | IndoBERTweet peringkat 5 dari 6, F1-macro 0.725. Terbaik: IndoBERT-base (baseline), 0.994 |
| Memakai **Emoji2Vec** | Tidak ada Emoji2Vec di notebook. Emoji dicocokkan ke leksikon lalu digabung dengan OR bitwise |
| Memakai **leksikon emoticon** | Tidak ada leksikon emoticon. `_norm()` justru menghapus emoticon lewat `[^a-z0-9\s]` |
| Prediksi disimulasikan di sisi browser | Antarmuka sekarang memanggil model sungguhan lewat `POST /api/predict` |

Antarmuka yang berlaku ada di `frontend/`, dijalankan oleh `backend/app.py`.
Penjelasan lengkap koreksinya ada di `README.md` utama, bagian
“Catatan untuk sidang”.
