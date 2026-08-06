"""
SIPEMO — uji bahwa setiap angka yang ditampilkan UI berasal dari notebook.

`backend/research.py` adalah satu-satunya sumber angka untuk tab **Evaluasi
Model** dan **Dataset & Metode** (frontend tidak menyimpan angka penelitian sama
sekali — lihat catatan di bawah). Karena angka di sana diketik manusia, angka itu
bisa saja salah salin. Skrip ini membandingkan setiap nilainya dengan output
tersimpan di `multilabel_bert_comparison.ipynb`.

Yang diperiksa:
  * 6 model x 5 metrik agregat + 4 F1 per label, presisi penuh   (cell 9)
  * 6 model x 3 epoch loss & val F1-macro + threshold             (cell 9)
  * hf_id tiap model                                              (cell 6 source)
  * statistik dataset & split                                     (cell 3, 4, 5)
  * hyperparameter & versi runtime                                (cell 1, 2)
  * best_result() harus memilih model yang sama dengan cell 14    (cell 14)

Tanpa dependensi di luar pustaka standar, jadi bisa dijalankan kapan saja:

    .venv\\Scripts\\python backend/verify_research.py

Catatan soal frontend: `frontend/js/` tidak memuat satu pun angka penelitian.
Semua berasal dari `/api/research`; satu-satunya angka turunan di sana adalah
persentase korpus terpakai (`rows_labeled / rows_after_prep`), yang dihitung dari
nilai API. Diperiksa oleh `check_frontend_has_no_numbers()` di bawah.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

import research as R  # noqa: E402

NOTEBOOK = PROJECT_DIR / "multilabel_bert_comparison.ipynb"

results: list[tuple[bool, str, str, str]] = []


def check(name: str, got, want) -> None:
    ok = repr(got) == repr(want) if isinstance(got, float) else got == want
    results.append((bool(ok), name, repr(got), repr(want)))


def cells() -> tuple[list[str], list[str]]:
    """(source tiap cell, output stream tiap cell)."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src, out = [], []
    for c in nb["cells"]:
        src.append("".join(c["source"]))
        txt = ""
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                txt += "".join(o.get("text", []))
            elif o.get("output_type") == "execute_result":
                txt += "".join(o.get("data", {}).get("text/plain", []))
        out.append(txt)
    return src, out


def notebook_model_dicts(c9: str) -> dict[str, dict]:
    """Semua dict hasil yang dicetak `print(r)` di akhir train_one()."""
    found = {}
    for m in re.finditer(r"\{'model': '([^']+)'", c9):
        i = m.start()
        found[m.group(1)] = ast.literal_eval(c9[i : c9.find("\n", i)])
    return found


def notebook_epochs(c9: str) -> dict[str, list[dict]]:
    """Baris '[nama] ep=N loss=.. val_F1macro=.. thr=..' per model."""
    out: dict[str, list[dict]] = {}
    for name, ep, loss, f1, thr in re.findall(
        r"\[([^\]]+)\] ep=(\d+) loss=([\d.]+) val_F1macro=([\d.]+) thr=([\d.]+)", c9
    ):
        out.setdefault(name, []).append(
            {"ep": int(ep), "loss": float(loss), "val_f1": float(f1), "thr": float(thr)}
        )
    return out


def check_frontend_has_no_numbers() -> None:
    """
    Pastikan frontend tidak menyimpan angka penelitian sendiri.

    Termasuk angka TURUNAN. Versi awal pemeriksaan ini hanya mencari metrik dan
    jumlah baris, sehingga lolos dari judul banner '88,8% korpus tidak terpakai'
    yang ternyata diketik langsung di views.js. Angka turunan sama rawannya jadi
    basi: kalau dataset berubah, angka itu tidak ikut berubah.
    """
    needles = [
        # metrik & statistik dataset
        "0.994", "0.9945", "0.9789", "0.9934", "2188", "19491", "1492",
        "2.188", "19.491",
        # angka turunan yang seharusnya dihitung dari nilai API
        "88,8", "88.8", "11,2", "11.2", "17303", "17.303",
    ]
    hits = []
    for js in sorted((PROJECT_DIR / "frontend" / "js").glob("*.js")):
        text = js.read_text(encoding="utf-8", errors="ignore")
        for n in needles:
            if n in text:
                hits.append(f"{js.name}:{n}")
    check("frontend tidak meng-hardcode angka penelitian/turunan", hits, [])


def check_frontend_labeling_claims() -> None:
    """
    Cari klaim pelabelan yang sudah terbukti keliru di prosa frontend.

    Pelabelan dataset memakai pencocokan kata kunci `_lex_one()`; jalur model
    neural tidak pernah menghasilkan label. Karena itu kalimat yang menjelaskan
    pembuangan baris lewat 'skor >= 0.30' atau menyebut model pelabel neural
    sebagai sumber label adalah klaim yang salah, dan tidak boleh ada di UI.
    """
    bad = [
        ("skor ≥ 0.30", "ambang skor tidak pernah dipakai — pelabelan lewat kata kunci"),
        ("skor >= 0.30", "ambang skor tidak pernah dipakai — pelabelan lewat kata kunci"),
        ("top-k=2", "top-k tidak pernah berlaku; keluaran model pelabel dibuang"),
    ]
    hits = []
    for js in sorted((PROJECT_DIR / "frontend" / "js").glob("*.js")):
        text = js.read_text(encoding="utf-8", errors="ignore")
        for needle, why in bad:
            if needle in text:
                hits.append(f"{js.name}: '{needle}' ({why})")
    check("frontend tidak memuat klaim pelabelan yang keliru", hits, [])


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"ERROR: notebook tidak ditemukan: {NOTEBOOK}", file=sys.stderr)
        return 1

    src, out = cells()
    c1s, c2s, c6s = src[1], src[2], src[6]
    c1o, c3o, c4o, c5o, c9o, c14o = out[1], out[3], out[4], out[5], out[9], out[14]

    nb_models = notebook_model_dicts(c9o)
    nb_epochs = notebook_epochs(c9o)
    nb_hf = ast.literal_eval(c6s[c6s.find("{") : c6s.rfind("}") + 1])

    print(f"model dalam output notebook : {len(nb_models)}")
    print(f"model dalam research.py     : {len(R.MODEL_RESULTS)}")

    # ---- 1. hasil per model ---------------------------------------------
    check("jumlah model", len(R.MODEL_RESULTS), len(nb_models))
    for r in R.MODEL_RESULTS:
        name = r["model"]
        nb = nb_models.get(name)
        if nb is None:
            check(f"{name}: ada di output notebook", "tidak ada", "ada")
            continue
        # Threshold dibandingkan pada 2 desimal, sengaja — bukan untuk
        # melonggarkan pemeriksaan. Cell 8 menyapu np.arange(0.20, 0.55, 0.05),
        # yang menghasilkan 0.39999999999999997 dan 0.44999999999999996 alih-alih
        # 0.40 dan 0.45 (0.2/0.25/0.3/0.35 kebetulan tepat). Selisihnya ~5e-17.
        # Tabel cell 11 — rujukan yang masuk skripsi — mencetak 0.40 dan 0.45,
        # dan research.py mengikuti tabel itu. Membandingkan pada presisi penuh
        # berarti menuntut research.py menyalin artefak float, bukan angka
        # penelitian. Dua desimal adalah presisi yang notebook sendiri tampilkan.
        check(f"{name} · threshold (2 desimal)", round(float(r["threshold"]), 2), round(float(nb["threshold"]), 2))
        for k in ("F1_micro", "F1_macro", "F1_samples", "Hamming", "SubsetAcc"):
            check(f"{name} · {k}", float(r[k]), float(nb[k]))
        check(
            f"{name} · per_label_F1",
            [float(x) for x in r["per_label_F1"]],
            [float(x) for x in nb["per_label_F1"]],
        )
        check(f"{name} · hf_id", r["hf_id"], nb_hf.get(name))

        eps = nb_epochs.get(name, [])
        check(f"{name} · train_loss", [float(x) for x in r["train_loss"]], [e["loss"] for e in eps])
        check(
            f"{name} · val_f1_macro",
            [float(x) for x in r["val_f1_macro"]],
            [e["val_f1"] for e in eps],
        )

    # ---- 2. dataset ------------------------------------------------------
    D = R.DATASET
    m = re.search(r"rows after prep:\s*(\d+)\s*\|\s*emoji rate:\s*([\d.]+)%", c3o)
    check("dataset · rows_after_prep", D["rows_after_prep"], int(m.group(1)))
    check("dataset · emoji_rate_pct", float(D["emoji_rate_pct"]), float(m.group(2)))

    m = re.search(r"\|\s*rows=(\d+)\s*\(([\d.]+)s\)", c3o)
    check("dataset · rows_labeled", D["rows_labeled"], int(m.group(1)))
    check("dataset · labeling_seconds", float(D["labeling_seconds"]), float(m.group(2)))

    for c in R.__dict__.get("LABEL_COLS", ["senang", "sedih", "marah", "takut"]):
        want = int(re.search(rf"^\s+{c}:\s*(\d+)/", c3o, re.M).group(1))
        check(f"dataset · label_counts[{c}]", D["label_counts"][c], want)

    m = re.search(r"cardinality mean:\s*([\d.]+)\s*max:\s*(\d+)", c4o)
    check("dataset · cardinality_mean", float(D["cardinality_mean"]), float(m.group(1)))
    check("dataset · cardinality_max", D["cardinality_max"], int(m.group(2)))
    check(
        "dataset · avg_labels_per_tweet",
        float(D["avg_labels_per_tweet"]),
        float(re.search(r"avg label/tweet:\s*([\d.]+)", c4o).group(1)),
    )

    combos = re.search(r"top combo:\s*(\[.*?\])\s*\n", c4o, re.S).group(1)
    want_combos = [
        {"combo": list(t), "count": n}
        for t, n in ast.literal_eval(re.sub(r"np\.int32\((\d+)\)", r"\1", combos))
    ]
    check("dataset · top_combos", D["top_combos"], want_combos)

    m = re.search(r"text len p50/p95:\s*(\d+)\s+(\d+)", c4o)
    check("dataset · text_len_p50", D["text_len_p50"], int(m.group(1)))
    check("dataset · text_len_p95", D["text_len_p95"], int(m.group(2)))

    check("dataset · split", D["split"], ast.literal_eval(re.search(r"\{'train':.*?\}", c5o).group(0)))

    # ---- 3. hyperparameter & runtime -------------------------------------
    T = R.TRAINING
    m = re.search(r"MAX_LEN,\s*BATCH,\s*EPOCHS,\s*LR\s*=\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.e-]+)", c1s)
    check("training · max_len", T["max_len"], int(m.group(1)))
    check("training · batch_size", T["batch_size"], int(m.group(2)))
    check("training · epochs", T["epochs"], int(m.group(3)))
    check("training · learning_rate", float(T["learning_rate"]), float(m.group(4)))
    check("training · seed", T["seed"], int(re.search(r"^SEED\s*=\s*(\d+)", c1s, re.M).group(1)))

    m = re.search(r"Device:\s*(\S+)\s+torch:\s*(\S+)\s+py:\s*(\S+)", c1o)
    check("training · device", T["device"], m.group(1))
    check("training · torch", T["torch"], m.group(2))
    check("training · python", T["python"], m.group(3))
    check(
        "training · transformers",
        T["transformers"],
        re.search(r"transformers==([\d.]+)", c2s).group(1),
    )

    # ---- 4. pemilihan model terbaik --------------------------------------
    m = re.search(r"best model by F1-macro:\s*(.+?)\s*\(F1_macro=", c14o)
    check("best_result() sama dengan cell 14", R.best_result()["model"], m.group(1))

    # ---- 5. frontend ------------------------------------------------------
    check_frontend_has_no_numbers()
    check_frontend_labeling_claims()

    # ---- ringkasan --------------------------------------------------------
    fails = [r for r in results if not r[0]]
    width = max(len(n) for _, n, _, _ in results) + 2
    print("\n" + "=" * 78)
    for ok, name, got, want in results:
        if ok:
            print(f"  OK    {name:<{width}}")
        else:
            print(f"  BEDA  {name:<{width}}\n        research.py: {got}\n        notebook   : {want}")
    print("=" * 78)
    print(
        f"{len(results) - len(fails)}/{len(results)} nilai cocok"
        + ("" if fails else " — semua angka yang tampil di UI terlacak ke output notebook.")
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
