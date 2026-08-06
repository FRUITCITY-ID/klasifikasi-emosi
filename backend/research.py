"""
SIPEMO — hasil penelitian (beku / frozen).

Seluruh angka di file ini disalin langsung dari output eksekusi notebook
`multilabel_bert_comparison.ipynb`. Tidak ada angka yang dihitung ulang,
diestimasi, atau dikarang. Sumber tiap blok ditulis di komentar.

Kalau notebook di-run ulang, perbarui file ini dari output yang baru.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Dataset & pelabelan  (output cell 3, 4, 5)
# --------------------------------------------------------------------------
DATASET = {
    "source_file": "tugas-akhir_cleaned.xlsx",
    "rows_after_prep": 19491,          # 'rows after prep: 19491'
    "emoji_rate_pct": 26.3,            # 'emoji rate: 26.3%'
    # 'mode=HF top-k=2 | rows=2188' — penanda 'HF' pada baris itu menyesatkan;
    # lihat label_provenance di bawah. Jumlah barisnya sendiri benar.
    "rows_labeled": 2188,
    # Waktu yang dihabiskan cell 3 menjalankan model pelabel pada 19.491 baris.
    # Keluarannya dibuang seluruhnya, jadi durasi ini tidak memengaruhi label.
    "labeling_seconds": 3245.6,
    "label_counts": {                  # cell 3
        "senang": 1492,
        "sedih": 297,
        "marah": 170,
        "takut": 333,
    },
    "split": {"train": 1750, "val": 219, "test": 219, "total": 2188},  # cell 5
    "cardinality_mean": 1.0475319926873858,   # cell 4
    "cardinality_max": 4,
    "avg_labels_per_tweet": 1.05,
    "top_combos": [                    # cell 4 — (senang, sedih, marah, takut)
        {"combo": [1, 0, 0, 0], "count": 1429},
        {"combo": [0, 0, 0, 1], "count": 297},
        {"combo": [0, 1, 0, 0], "count": 238},
        {"combo": [0, 0, 1, 0], "count": 129},
        {"combo": [1, 1, 0, 0], "count": 30},
    ],
    "text_len_p50": 25,                # cell 4
    "text_len_p95": 46,
    # Pelabel yang BENAR-BENAR dipakai adalah leksikon kata kunci di cell 3,
    # bukan model neural. Diverifikasi dengan reproduksi penuh: menjalankan
    # HANYA jalur leksikon menghasilkan 19.491 -> 2.188 baris dengan sebaran
    # 1492/297/170/333 dan emoji rate 26,3% — tujuh angka, semuanya identik
    # dengan output cell 3. Kalau jalur HF pernah menyumbang satu label saja,
    # angka-angka itu tidak akan cocok. Lihat training/build_corpus.py.
    "labeler_model": "Leksikon kata kunci 4 kelas (_LEX, notebook cell 3)",
    "labeler_strategy": (
        "Cocokkan kata pada text_norm dengan 4 himpunan kata kunci; "
        "baris tanpa kecocokan dibuang (19.491 -> 2.188)"
    ),
    "label_provenance": (
        "Silver label — dihasilkan aturan kata kunci deterministik, bukan "
        "anotasi manusia dan bukan model pelabel neural. Cell 3 memang memuat "
        "StevenLimcorn/indonesian-roberta-base-emotion-classifier lalu "
        "menjalankannya pada tiap baris, tetapi keluarannya tidak pernah "
        "terpakai: _HF_MAP berkunci 'Senang/Sedih/Marah/Takut' sedangkan model "
        "itu mengeluarkan 'sadness/anger/love/fear/happy', sehingga "
        "_HF_MAP.get() selalu None, bins tetap [0,0,0,0], syarat sum(b) > 0 "
        "selalu gagal, dan tiap baris jatuh ke _lex_one(). Implikasinya: "
        "F1-macro 0.994 mengukur seberapa baik IndoBERT menirukan aturan kata "
        "kunci — aturan yang membaca teks yang sama persis dengan modelnya."
    ),
    # Entri emoji di dalam _LEX ('😊', '😭', ...) tidak pernah aktif saat
    # pelabelan: label dihitung atas text_norm, dan _norm() sudah menghapus
    # seluruh emoji lewat regex [^a-z0-9\s]. Dihitung ulang atas 2.188 baris:
    # nol kecocokan. Emoji baru berperan saat inferensi (fusi OR di cell 15).
    "lexicon_top_triggers": {          # kata pemicu terbanyak per label
        "senang": [["love", 418], ["good", 250], ["suka", 193], ["happy", 174]],
        "sedih": [["sedih", 126], ["galau", 86], ["kecewa", 30], ["kehilangan", 30]],
        "marah": [["emosi", 52], ["marah", 43], ["sebel", 17], ["anjir", 13]],
        "takut": [["anxious", 136], ["takut", 85], ["cemas", 35], ["scared", 26]],
    },
}

# --------------------------------------------------------------------------
# Hyperparameter training  (cell 1, 9)
# --------------------------------------------------------------------------
TRAINING = {
    "seed": 42,
    "max_len": 96,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "optimizer": "AdamW",
    "scheduler": "linear, warmup=0",
    "loss": "BCEWithLogits (problem_type='multi_label_classification')",
    "threshold_search": "grid 0.20–0.50 step 0.05, dipilih pada validation set (F1-macro)",
    "device": "cpu",
    "torch": "2.13.0+cpu",
    "transformers": "4.44.0",
    "python": "3.10.0",
}

# --------------------------------------------------------------------------
# Hasil per model  (output cell 9 & tabel cell 11)
# per_label_F1 berurutan: senang, sedih, marah, takut
# --------------------------------------------------------------------------
MODEL_RESULTS = [
    {
        "model": "IndoBERT-base (baseline)",
        "hf_id": "indobenchmark/indobert-base-p1",
        "threshold": 0.30,
        "F1_micro": 0.9933774834437086,
        "F1_macro": 0.9944570625135194,
        "F1_samples": 0.9954337899543378,
        "Hamming": 0.003424657534246575,
        "SubsetAcc": 0.9863013698630136,
        "per_label_F1": [0.9927536231884058, 1.0, 1.0, 0.9850746268656716],
        "train_loss": [0.2955, 0.0779, 0.0448],
        "val_f1_macro": [0.8687, 0.9620, 0.9789],
    },
    {
        "model": "IndoBERT-IndoNLU",
        "hf_id": "indobenchmark/indobert-base-p2",
        "threshold": 0.45,
        "F1_micro": 0.9844097995545658,
        "F1_macro": 0.977911723079772,
        "F1_samples": 0.9878234398782343,
        "Hamming": 0.007990867579908675,
        "SubsetAcc": 0.9726027397260274,
        "per_label_F1": [
            0.9890909090909091,
            0.9873417721518988,
            0.9655172413793104,
            0.9696969696969697,
        ],
        "train_loss": [0.2902, 0.0851, 0.0492],
        "val_f1_macro": [0.8649, 0.9694, 0.9880],
    },
    {
        "model": "mBERT-base",
        "hf_id": "bert-base-multilingual-cased",
        "threshold": 0.30,
        "F1_micro": 0.8489795918367347,
        "F1_macro": 0.7416654180401558,
        "F1_samples": 0.8759512937595129,
        "Hamming": 0.08447488584474885,
        "SubsetAcc": 0.7442922374429224,
        "per_label_F1": [0.9507042253521126, 0.7659574468085106, 0.375, 0.875],
        "train_loss": [0.4096, 0.2441, 0.1607],
        "val_f1_macro": [0.3773, 0.7420, 0.7816],
    },
    {
        "model": "IndoBERT-IndoLEM",
        "hf_id": "indolem/indobert-base-uncased",
        "threshold": 0.25,
        "F1_micro": 0.8559837728194726,
        "F1_macro": 0.7397899859790104,
        "F1_samples": 0.882648401826484,
        "Hamming": 0.08105022831050228,
        "SubsetAcc": 0.771689497716895,
        "per_label_F1": [
            0.9407665505226481,
            0.7916666666666666,
            0.3888888888888889,
            0.8378378378378378,
        ],
        "train_loss": [0.4201, 0.2978, 0.1715],
        "val_f1_macro": [0.4331, 0.7296, 0.7582],
    },
    {
        "model": "IndoBERTweet",
        "hf_id": "indolem/indobertweet-base-uncased",
        "threshold": 0.40,
        "F1_micro": 0.874439461883408,
        "F1_macro": 0.7249086499086499,
        "F1_samples": 0.867579908675799,
        "Hamming": 0.0639269406392694,
        "SubsetAcc": 0.8310502283105022,
        "per_label_F1": [
            0.9370629370629371,
            0.7837837837837838,
            0.3,
            0.8787878787878788,
        ],
        "train_loss": [0.3906, 0.2173, 0.1230],
        "val_f1_macro": [0.6434, 0.8257, 0.8658],
    },
    {
        "model": "XLM-RoBERTa-base",
        "hf_id": "xlm-roberta-base",
        "threshold": 0.45,
        "F1_micro": 0.8746928746928747,
        "F1_macro": 0.6416221828214925,
        "F1_samples": 0.8051750380517503,
        "Hamming": 0.05821917808219178,
        "SubsetAcc": 0.7899543378995434,
        "per_label_F1": [0.9548872180451128, 0.6885245901639344, 0.0, 0.9230769230769231],
        "train_loss": [0.4360, 0.2815, 0.2063],
        "val_f1_macro": [0.5520, 0.6264, 0.6746],
    },
]


def best_result() -> dict:
    """Model dengan F1-macro tertinggi — sama dengan pemilihan di cell 14."""
    return max(MODEL_RESULTS, key=lambda r: r["F1_macro"])


def result_for(model_name: str) -> dict | None:
    for r in MODEL_RESULTS:
        if r["model"] == model_name:
            return r
    return None


# --------------------------------------------------------------------------
# Catatan metodologis yang perlu tampil apa adanya di UI.
# --------------------------------------------------------------------------
CAVEATS = [
    {
        "level": "high",
        "title": "Label berasal dari aturan kata kunci, bukan anotasi manusia",
        "body": (
            "Label latih dihasilkan leksikon 4 kelas di cell 3 — bukan anotasi "
            "manusia. Model pelabel neural yang dimuat cell 3 tidak pernah "
            "menyumbang label karena nama kelasnya tidak cocok dengan _HF_MAP "
            "(lihat catatan 'Pelabel yang sebenarnya'). Karena aturan itu "
            "deterministik dan membaca teks yang sama dengan modelnya, F1-macro "
            "0.994 terutama menunjukkan IndoBERT sanggup menghafal aturan kata "
            "kunci — bukan bukti akurasi terhadap emosi sebenarnya."
        ),
    },
    {
        "level": "high",
        "title": "Pelabel yang sebenarnya: _lex_one(), bukan RoBERTa",
        "body": (
            "Cell 3 memuat StevenLimcorn/indonesian-roberta-base-emotion-classifier "
            "dan menjalankannya pada 19.491 baris (3.246 detik), tetapi "
            "keluarannya dibuang: _HF_MAP berkunci 'Senang/Sedih/Marah/Takut' "
            "sedangkan model itu mengeluarkan 'sadness/anger/love/fear/happy'. "
            "_HF_MAP.get() selalu None → bins tetap [0,0,0,0] → syarat "
            "sum(b) > 0 selalu gagal → tiap baris memakai _lex_one(). Notebook "
            "tetap mencetak 'mode=HF top-k=2' karena penanda itu hanya mengecek "
            "apakah pipeline berhasil dimuat. Diverifikasi lewat reproduksi: "
            "jalur leksikon saja menghasilkan tujuh angka yang identik dengan "
            "output cell 3 (19.491 → 2.188; 1492/297/170/333; emoji 26,3%)."
        ),
    },
    {
        "level": "high",
        "title": "Tidak ada kelas netral — teks netral jatuh ke senang",
        "body": (
            "Baris yang tidak memuat satu pun kata kunci leksikon dibuang saat "
            "pelabelan (19.491 menjadi 2.188), sehingga model tidak pernah "
            "melihat contoh 'tidak ada emosi' dan tidak punya kelas netral "
            "untuk ditempati. Digabung dengan 1.492 dari 2.188 baris latih yang "
            "berlabel senang dan ambang hasil tuning yang hanya 0,30, kalimat "
            "netral cenderung keluar sebagai senang. Diukur pada 15 kalimat "
            "netral: 14 di antaranya dilabeli senang. Teks yang seluruh "
            "karakternya terhapus _norm() — misalnya '...' — juga tetap "
            "menghasilkan senang, karena tokenizer hanya menerima [CLS] [SEP] "
            "dan model mengembalikan prior-nya. Angka di tab Evaluasi tetap "
            "sahih untuk test set-nya, tetapi test set itu seluruhnya berisi "
            "teks yang emosinya eksplisit secara leksikal. Pengukuran ini bisa "
            "diulang dengan backend/probe_neutral.py."
        ),
    },
    {
        "level": "high",
        "title": "Kelas marah sangat kecil",
        "body": (
            "Hanya 170 dari 2.188 baris berlabel marah, sehingga di test set "
            "kelas ini sangat sedikit. XLM-RoBERTa mendapat F1 0.00 dan "
            "IndoBERTweet 0.30 pada label ini — F1-macro mereka jatuh terutama "
            "karena kelas marah."
        ),
    },
    {
        "level": "medium",
        "title": "Multi-label dalam praktiknya hampir single-label",
        "body": (
            "Rata-rata kardinalitas 1.05 label per tweet; 1.429 dari 2.188 baris "
            "hanya berlabel senang. Kemampuan multi-label sistem jarang teruji "
            "pada data ini."
        ),
    },
    {
        "level": "medium",
        "title": "Split tanpa stratifikasi",
        "body": (
            "Cell 5 melakukan train_test_split acak tanpa stratify, sehingga "
            "distribusi kelas minoritas antar-split tidak dijamin seimbang."
        ),
    },
    {
        "level": "medium",
        "title": "Fusi terjadi di level keputusan, bukan level fitur",
        "body": (
            "Emoji tidak masuk ke encoder. Teks yang masuk BERT sudah bersih "
            "dari emoji (regex [^a-z0-9\\s] menghapusnya). Emoji dicocokkan "
            "terpisah ke leksikon, lalu digabung dengan OR bitwise: "
            "pred_fused = pred_bert | pred_emoji. Tidak ada Emoji2Vec dan tidak "
            "ada konkatenasi vektor."
        ),
    },
]
