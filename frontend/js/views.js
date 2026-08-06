/* ==========================================================================
   SIPEMO — render tampilan
   ========================================================================== */

import { hbar, miniLine, table, withTableView } from "./charts.js";
import { dec, int, pct, pctRaw, times } from "./format.js";

export const LABEL_COLOR = {
  senang: "var(--c-senang)",
  sedih: "var(--c-sedih)",
  marah: "var(--c-marah)",
  takut: "var(--c-takut)",
};

const f3 = (v) => (v === null || v === undefined ? "—" : dec(Number(v), 3));
const f4 = (v) => (v === null || v === undefined ? "—" : dec(Number(v), 4));

/* Helper pembuat elemen ringkas. */
export function h(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function")
      node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) node.setAttribute(k, "");
    else if (v !== false && v !== null && v !== undefined)
      node.setAttribute(k, String(v));
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return node;
}

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function icon(path, cls = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  if (cls) svg.setAttribute("class", cls);
  svg.innerHTML = path;
  return svg;
}

const ICON_ALERT = '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/>';
const ICON_INFO = '<circle cx="12" cy="12" r="9"/><path d="M12 16v-5M12 8.5v.01"/>';

/* --------------------------------------------------------------------------
   Header — chip status model
   -------------------------------------------------------------------------- */
export function renderStatusChip(chip, health) {
  const m = health?.model;
  chip.innerHTML = "";
  let state = "loading";
  let text = "Memuat model…";

  if (m) {
    if (m.status === "ready") {
      state = "ready";
      text = `${m.model_name} · σ ≥ ${dec(m.threshold, 2)}`;
    } else if (m.status === "no_checkpoint") {
      state = "error";
      text = "Checkpoint tidak ditemukan";
    } else if (m.status === "mismatch") {
      state = "error";
      text = "Checkpoint tidak cocok";
    } else {
      state = "error";
      text = "Model tidak siap";
    }
  }
  chip.dataset.state = state;
  chip.append(h("span", { class: "dot" }), h("span", { text }));
}

/* Banner penjelas ketika model belum siap. */
export function renderModelBanner(host, health) {
  host.innerHTML = "";
  const m = health?.model;
  if (!m || m.status === "ready") {
    if (m && m.warnings?.length) {
      host.appendChild(
        h("div", { class: "banner", "data-level": "warning" },
          icon(ICON_ALERT, "banner-icon"),
          h("div", {},
            h("div", { class: "banner-title", text: "Model dimuat dengan peringatan" }),
            h("div", { class: "banner-body", text: m.warnings.join("\n\n") })
          )
        )
      );
    }
    return;
  }

  const titles = {
    no_checkpoint: "Checkpoint belum tersedia — prediksi dinonaktifkan",
    mismatch: "Checkpoint tidak cocok dengan IndoBERT-base — prediksi dinonaktifkan",
    error: "Model gagal dimuat",
  };

  host.appendChild(
    h("div", { class: "banner", "data-level": "error" },
      icon(ICON_ALERT, "banner-icon"),
      h("div", {},
        h("div", { class: "banner-title", text: titles[m.status] || "Model tidak siap" }),
        h("div", { class: "banner-body", text: m.message }),
        h("div", { class: "row", style: "margin-top:12px" },
          h("button", { class: "btn btn-quiet btn-sm", id: "retryLoad", text: "Muat ulang model" }),
          h("span", { class: "hint", text: `Checkpoint: ${m.checkpoint_path}` })
        )
      )
    )
  );
}

/* --------------------------------------------------------------------------
   Tampilan Analisis
   -------------------------------------------------------------------------- */
export function renderResult(host, r) {
  host.innerHTML = "";
  const labels = r.labels;

  /* --- Peringatan teks habis setelah normalisasi ------------------------ */
  /* Kalau _norm() menyisakan string kosong, tokenizer hanya menerima
     [CLS] [SEP] dan model mengembalikan prior-nya — label yang muncul tidak
     ada hubungannya dengan apa yang diketik. Kasus ini harus dikatakan, bukan
     dibiarkan tampil sebagai prediksi biasa. */
  if (!r.norm) {
    host.appendChild(
      h("div", { class: "banner", "data-level": "warning" },
        icon(ICON_ALERT, "banner-icon"),
        h("div", {},
          h("div", { class: "banner-title", text: "Tidak ada teks tersisa untuk dibaca model" }),
          h("div", { class: "banner-body", text:
            "Seluruh karakter terhapus tahap pembersihan, sehingga tokenizer hanya " +
            "menerima [CLS] dan [SEP]. Angka di bawah adalah kecenderungan bawaan " +
            "model pada masukan kosong, bukan pembacaan atas teks Anda. Label dari " +
            "jalur emoji (bila ada) tetap sahih karena dicocokkan ke teks mentah." })
        )
      )
    );
  }

  /* --- Verdict ---------------------------------------------------------- */
  const tags = r.active.length
    ? r.active.map((c) =>
        h("span", { class: "label-tag" },
          h("span", { class: "swatch", style: `background:${LABEL_COLOR[c]}` }),
          c
        )
      )
    : [h("span", { class: "label-tag muted" }, "Tidak ada label melewati threshold")];

  host.appendChild(
    h("div", { class: "verdict" },
      h("div", {},
        h("div", { class: "eyebrow", text: `Label aktif — fusi BERT ∨ emoji, σ ≥ ${dec(r.threshold, 2)}` }),
        h("div", { class: "verdict-labels", style: "margin-top:8px" }, tags)
      ),
      h("div", { class: "verdict-meta" },
        h("b", { class: "tnum", text: String(r.active.length) }),
        `dari ${labels.length} kategori · ${dec(r.latency_ms, 1)} ms`
      )
    )
  );

  /* --- Chart probabilitas ------------------------------------------------ */
  const data = labels.map((c) => ({
    key: c,
    label: c,
    value: r.probs[c],
    color: LABEL_COLOR[c],
  }));

  const chart = hbar({
    data,
    max: 1,
    threshold: r.threshold,
    thresholdLabel: `threshold ${dec(r.threshold, 2)}`,
    format: (v) => dec(v, 3),
    tickFormat: (v) => dec(v, 2),
    tooltipHtml: (d) =>
      `<div class="tt-title">${d.label}</div>` +
      `<div class="tt-row">sigmoid: ${dec(d.value, 4)}</div>` +
      `<div class="tt-row">logit: ${dec(r.logits[d.key], 3)}</div>` +
      `<div class="tt-row">BERT: ${r.pred_bert[d.key] ? "aktif" : "tidak"} · emoji: ${
        r.pred_emoji[d.key] ? "aktif" : "tidak"
      }</div>`,
  });

  const probTable = table(
    [
      { label: "Emosi", get: (d) => `<span style="display:inline-flex;align-items:center;gap:8px"><span style="width:8px;height:8px;border-radius:50%;background:${LABEL_COLOR[d.key]};display:inline-block"></span>${d.label}</span>` },
      { label: "Sigmoid", get: (d) => f4(d.value) },
      { label: "Logit", get: (d) => f3(r.logits[d.key]) },
      { label: "BERT", get: (d) => String(r.pred_bert[d.key]) },
      { label: "Emoji", get: (d) => String(r.pred_emoji[d.key]) },
      { label: "Fusi", get: (d) => String(r.pred_fused[d.key]) },
    ],
    data
  );

  const pair = withTableView(chart, probTable);
  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "chart-head" },
        h("div", {},
          h("div", { class: "card-title", text: "Probabilitas per emosi" }),
          h("div", { class: "card-sub", text: "Keluaran sigmoid — tiap label diprediksi independen, bukan softmax" })
        ),
        h("div", { class: "spacer" }),
        pair._toggle
      ),
      pair
    )
  );

  /* --- Matriks fusi ----------------------------------------------------- */
  const rows = labels.map((c) =>
    h("tr", {},
      h("th", { scope: "row" },
        h("span", { class: "lab-cell" },
          h("span", { class: "swatch", style: `background:${LABEL_COLOR[c]}` }),
          c
        )
      ),
      h("td", { class: "prob-cell tnum", text: dec(r.probs[c], 4) }),
      h("td", {}, h("span", { class: "bit", "data-on": String(r.pred_bert[c]), text: String(r.pred_bert[c]) })),
      h("td", {}, h("span", { class: "bit", "data-on": String(r.pred_emoji[c]), text: String(r.pred_emoji[c]) })),
      h("td", {}, h("span", { class: "bit", "data-fused": String(r.pred_fused[c]), text: String(r.pred_fused[c]) }))
    )
  );

  const fusionTable = h("table", { class: "fusion" },
    h("thead", {},
      h("tr", {},
        h("th", { scope: "col", text: "Emosi" }),
        h("th", { scope: "col", text: "σ" }),
        h("th", { scope: "col", text: "BERT" }),
        h("th", { scope: "col", text: "Emoji" }),
        h("th", { scope: "col", text: "Fusi" })
      )
    ),
    h("tbody", {}, rows)
  );

  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "card-head" },
        h("div", {},
          h("div", { class: "card-title", text: "Fusi level keputusan" }),
          h("div", { class: "card-sub", text: "Emoji tidak masuk encoder — keduanya digabung setelah masing-masing menghasilkan keputusan biner" })
        )
      ),
      fusionTable,
      /* Blok ini menampilkan ekspresi Python apa adanya, jadi angkanya sengaja
         TIDAK dilewatkan format id-ID: 0.30 di sini literal kode, bukan angka
         bacaan. Satu-satunya tempat titik desimal masih dipakai. */
      h("div", { class: "formula", text:
        `bin_bert  = sigmoid(logits) >= ${r.threshold.toFixed(2)}\n` +
        `bin_emoji = label_emoji_set(EMOJI_RX.findall(teks_mentah))\n` +
        `bin_fused = bin_bert | bin_emoji` })
    )
  );

  /* --- Jejak pipeline --------------------------------------------------- */
  host.appendChild(renderPipeline(r));
}

function renderPipeline(r) {
  const p = r.pipeline;

  const steps = p.steps.map((s, i) =>
    h("div", { class: "pstep", "data-changed": String(s.changed) },
      h("div", {}, h("span", { class: "pstep-idx", text: String(i + 1) })),
      h("div", {},
        h("div", { class: "pstep-name" },
          s.label,
          h("span", { class: "pstep-rule", text: s.detail }),
          !s.changed && h("span", { class: "pstep-noop", text: "· tidak ada perubahan" })
        ),
        h("div", { class: "pstep-text", text: s.text })
      )
    )
  );

  const slang = p.slang_changes.length
    ? h("div", { class: "pills", style: "margin-top:8px" },
        p.slang_changes.map((c) =>
          h("span", { class: "pill", "data-kind": "sub", text: `${c.from} → ${c.to}` })
        )
      )
    : h("div", { class: "empty-note", text: "Tidak ada kata yang tergantikan kamus slang." });

  /* Wordpiece dari tokenizer IndoBERT */
  const pieces = h("div", { class: "pills" },
    p.wordpieces.map((tok) => {
      const special = /^\[.*\]$/.test(tok);
      const cont = tok.startsWith("##");
      return h("span", {
        class: "pill",
        "data-kind": special ? "special" : cont ? "cont" : "",
        text: tok,
      });
    })
  );

  const emojiBlock = p && r.emoji.found.length
    ? h("div", { class: "glyph-map" },
        r.emoji.matches.map((m) =>
          h("span", { class: "glyph-item", "data-hit": String(m.labels.length > 0) },
            h("span", { class: "g", text: m.emoji }),
            h("span", { class: "arrow", text: "→" }),
            h("span", { class: "lab", text: m.labels.length ? m.labels.join(", ") : "tidak ada di leksikon" })
          )
        )
      )
    : h("div", { class: "empty-note", text: "Tidak ada karakter yang cocok dengan EMOJI_RX pada teks mentah." });

  return h("div", { class: "card" },
    h("div", { class: "card-head" },
      h("div", {},
        h("div", { class: "card-title", text: "Jejak pipeline" }),
        h("div", { class: "card-sub", text: "Setiap tahap sama persis dengan _norm() dan predict_one() pada notebook" })
      )
    ),
    h("div", { class: "grid-2" },
      h("div", {},
        h("div", { class: "eyebrow", style: "margin-bottom:10px", text: "A · Jalur teks — pembersihan" }),
        h("div", { class: "pipeline" }, steps)
      ),
      h("div", {},
        h("div", { class: "eyebrow", style: "margin-bottom:10px", text: "B · Substitusi slang" }),
        slang,
        h("div", { class: "eyebrow", style: "margin:20px 0 10px", text:
          `C · Tokenisasi wordpiece — ${p.n_wordpieces}/${p.max_len} token${p.truncated ? " (terpotong)" : ""}` }),
        pieces,
        h("div", { class: "eyebrow", style: "margin:20px 0 10px", text: "D · Jalur emoji — leksikon" }),
        emojiBlock
      )
    )
  );
}

/* --------------------------------------------------------------------------
   Tampilan Evaluasi
   -------------------------------------------------------------------------- */
const METRICS = [
  { key: "F1_macro", label: "F1-macro", higher: true },
  { key: "F1_micro", label: "F1-micro", higher: true },
  { key: "F1_samples", label: "F1-samples", higher: true },
  { key: "SubsetAcc", label: "Subset Accuracy", higher: true },
  { key: "Hamming", label: "Hamming Loss", higher: false },
];

export function renderEvaluation(host, R) {
  host.innerHTML = "";
  const served = R.served_result;
  const results = R.results;

  /* --- Tile model terbaik ------------------------------------------------ */
  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "card-head" },
        h("div", {},
          h("div", { class: "eyebrow", text: "Model yang dilayani sistem" }),
          h("div", { class: "card-title", style: "font-size:var(--text-lg);margin-top:4px", text: R.served_model }),
          h("div", { class: "card-sub mono", text: R.served_model_id })
        ),
        h("div", { class: "spacer" }),
        h("div", { style: "text-align:right" },
          h("div", { class: "hero-figure", text: f3(served.F1_macro) }),
          h("div", { class: "tile-note", text: "F1-macro pada test set" })
        )
      ),
      h("div", { class: "grid-4" },
        tile("F1-micro", f3(served.F1_micro)),
        tile("F1-samples", f3(served.F1_samples)),
        tile("Subset Accuracy", f3(served.SubsetAcc)),
        tile("Hamming Loss", f4(served.Hamming), "makin kecil makin baik")
      )
    )
  );

  /* --- Filter metrik + perbandingan model -------------------------------- */
  const filterRow = h("div", { class: "row", style: "margin-bottom:4px" },
    h("span", { class: "eyebrow", text: "Metrik" })
  );
  const select = h("select", {
    id: "metricSelect",
    class: "btn btn-quiet btn-sm",
    style: "padding-right:8px",
    "aria-label": "Pilih metrik",
  }, METRICS.map((m) => h("option", { value: m.key, text: m.label })));
  filterRow.appendChild(select);
  filterRow.appendChild(h("span", { class: "hint", text: "berlaku untuk grafik dan tabel di bawah" }));
  host.appendChild(filterRow);

  const compareCard = h("div", { class: "card" });
  host.appendChild(compareCard);

  function drawCompare(metricKey) {
    const metric = METRICS.find((m) => m.key === metricKey);
    const sorted = [...results].sort((a, b) =>
      metric.higher ? b[metric.key] - a[metric.key] : a[metric.key] - b[metric.key]
    );
    const max = metric.higher ? 1 : Math.max(...results.map((r) => r.Hamming)) * 1.15;

    /* Emphasis: model yang dilayani penuh warna, sisanya diredam.
       Warna mengikuti entitas (model yang dilayani), bukan peringkat. */
    const data = sorted.map((r) => ({
      key: r.model,
      label: r.model.replace(" (baseline)", ""),
      value: r[metric.key],
      color: "var(--series-1)",
      dim: r.model !== R.served_model,
      raw: r,
    }));

    const chart = hbar({
      data,
      max,
      ticks: metric.higher ? [0, 0.25, 0.5, 0.75, 1] : [0, max / 2, max],
      gutterLeft: 150,
      format: (v) => dec(v, metric.higher ? 3 : 4),
      tickFormat: (v) => dec(v, 2),
      tooltipHtml: (d) =>
        `<div class="tt-title">${d.key}</div>` +
        `<div class="tt-row">${metric.label}: ${dec(d.value, 4)}</div>` +
        `<div class="tt-row">threshold: ${dec(d.raw.threshold, 2)}</div>` +
        `<div class="tt-row mono">${d.raw.hf_id}</div>`,
    });

    const tbl = table(
      METRICS.reduce(
        (cols, m) => {
          cols.push({ label: m.label, get: (r) => (m.higher ? f3(r[m.key]) : f4(r[m.key])) });
          return cols;
        },
        [
          {
            label: "Model",
            get: (r) =>
              esc(r.model) +
              (r.model === R.served_model ? '<span class="served-flag">DILAYANI</span>' : ""),
          },
          { label: "Threshold", get: (r) => dec(r.threshold, 2) },
        ]
      ),
      sorted,
      { isHighlighted: (r) => r.model === R.served_model }
    );

    const pair = withTableView(chart, tbl);
    compareCard.innerHTML = "";
    compareCard.append(
      h("div", { class: "chart-head" },
        h("div", {},
          h("div", { class: "card-title", text: `Perbandingan 6 model — ${metric.label}` }),
          h("div", { class: "card-sub", text: metric.higher
            ? "Makin panjang makin baik. Batang penuh warna = model yang dilayani sistem ini."
            : "Makin pendek makin baik. Batang penuh warna = model yang dilayani sistem ini." })
        ),
        h("div", { class: "spacer" }),
        pair._toggle
      ),
      pair,
      h("div", { class: "legend" },
        h("span", { class: "legend-item" },
          h("span", { class: "swatch", style: "background:var(--series-1)" }),
          R.served_model
        ),
        h("span", { class: "legend-item" },
          h("span", { class: "swatch", style: "background:var(--series-1);opacity:.4" }),
          "model pembanding"
        )
      )
    );
  }

  drawCompare("F1_macro");
  select.addEventListener("change", () => drawCompare(select.value));

  /* --- F1 per label, small multiples ------------------------------------- */
  const labels = R.labels;
  const facets = labels.map((c, li) => {
    const data = [...results]
      .sort((a, b) => b.per_label_F1[li] - a.per_label_F1[li])
      .map((r) => ({
        key: r.model,
        label: r.model.replace(" (baseline)", "").replace("-base", ""),
        value: r.per_label_F1[li],
        color: LABEL_COLOR[c],
        dim: r.model !== R.served_model,
      }));

    return h("div", {},
      h("div", { class: "sm-title" },
        h("span", { class: "swatch", style: `background:${LABEL_COLOR[c]}` }),
        c
      ),
      h("div", { class: "sm-sub", text: `${int(R.dataset.label_counts[c])} dari ${int(R.dataset.rows_labeled)} baris berlabel ini` }),
      hbar({
        data,
        max: 1,
        gutterLeft: 132,
        gutterRight: 48,
        ticks: [0, 0.5, 1],
        format: (v) => dec(v, 2),
        tickFormat: (v) => dec(v, 1),
        tooltipHtml: (d) =>
          `<div class="tt-title">${d.key}</div><div class="tt-row">F1 ${c}: ${dec(d.value, 4)}</div>`,
      })
    );
  });

  const perLabelTable = table(
    [
      { label: "Model", get: (r) => esc(r.model) },
      ...labels.map((c, li) => ({ label: `F1 ${c}`, get: (r) => f3(r.per_label_F1[li]) })),
    ],
    results,
    { isHighlighted: (r) => r.model === R.served_model }
  );

  const plPair = withTableView(h("div", { class: "small-multiples" }, facets), perLabelTable);
  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "chart-head" },
        h("div", {},
          h("div", { class: "card-title", text: "F1 per label" }),
          h("div", { class: "card-sub", text: "Satu panel per emosi. Kelas marah paling kecil datanya — di situ selisih antar-model paling tajam." })
        ),
        h("div", { class: "spacer" }),
        plPair._toggle
      ),
      plPair
    )
  );

  /* --- Kurva training ---------------------------------------------------- */
  const epochs = ["ep 1", "ep 2", "ep 3"];
  const curveCards = results.map((r) =>
    h("div", {},
      h("div", { class: "sm-title" },
        r.model.replace(" (baseline)", ""),
        r.model === R.served_model && h("span", { class: "served-flag", text: "DILAYANI" })
      ),
      h("div", { class: "sm-sub", text: `val F1-macro · threshold akhir ${dec(r.threshold, 2)}` }),
      miniLine({
        points: r.val_f1_macro,
        xLabels: epochs,
        yMin: 0,
        yMax: 1,
        color: "var(--series-1)",
        label: r.model,
        format: (v) => dec(v, 4),
      })
    )
  );

  const curveTable = table(
    [
      { label: "Model", get: (r) => esc(r.model) },
      ...epochs.map((e, i) => ({ label: `loss ${e}`, get: (r) => f4(r.train_loss[i]) })),
      ...epochs.map((e, i) => ({ label: `val F1 ${e}`, get: (r) => f4(r.val_f1_macro[i]) })),
    ],
    results,
    { isHighlighted: (r) => r.model === R.served_model }
  );

  const cPair = withTableView(
    h("div", { class: "small-multiples", style: "grid-template-columns:repeat(3,1fr)" }, curveCards),
    curveTable
  );
  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "chart-head" },
        h("div", {},
          h("div", { class: "card-title", text: "Validation F1-macro per epoch" }),
          h("div", { class: "card-sub", text: `${R.training.epochs} epoch, batch ${R.training.batch_size}, lr ${R.training.learning_rate}, ${R.training.optimizer}` })
        ),
        h("div", { class: "spacer" }),
        cPair._toggle
      ),
      cPair
    )
  );
}

function tile(label, value, note) {
  return h("div", { class: "tile" },
    h("div", { class: "tile-label", text: label }),
    h("div", { class: "tile-value", text: value }),
    note && h("div", { class: "tile-note", text: note })
  );
}

/* --------------------------------------------------------------------------
   Isi leksikon pelabel

   Setelah UI menyatakan "labelnya dari kata kunci, bukan anotasi manusia",
   pertanyaan berikutnya selalu "kata kuncinya apa saja?". Kartu ini
   menjawabnya langsung dari sumbernya: kata pemicu terbanyak berasal dari
   /api/research (dihitung ulang atas korpus oleh verify_reproduction.py), dan
   daftar kata lengkapnya dari /api/lexicon.
   -------------------------------------------------------------------------- */
function renderLexicon(R) {
  const D = R.dataset;
  const lex = R.lexicon;  /* boleh null kalau /api/lexicon gagal dimuat */
  const meta = R.label_meta || {};

  const kolom = R.labels.map((c) => {
    const triggers = (D.lexicon_top_triggers || {})[c] || [];

    /* Entri kata dan entri emoji dipisah: saat PELABELAN hanya entri kata yang
       pernah cocok, karena _norm() sudah menghapus emoji dari text_norm. */
    const semua = lex?.lexicon?.[c] || [];
    const emoji = new Set(lex?.emoji_lexicon?.[c] || []);
    const kata = semua.filter((w) => !emoji.has(w));

    return h("div", {},
      h("div", { class: "sm-title" },
        h("span", { class: "swatch", style: `background:${LABEL_COLOR[c]}` }),
        c
      ),
      meta[c]?.desc && h("div", { class: "sm-sub", text: meta[c].desc }),

      h("div", { class: "eyebrow", style: "margin:14px 0 8px", text: "Pemicu terbanyak" }),
      triggers.length
        ? h("div", { class: "pills" },
            triggers.map(([kataPemicu, jumlah]) =>
              h("span", { class: "pill", "data-kind": "sub", text: `${kataPemicu} · ${int(jumlah)}` })
            )
          )
        : h("div", { class: "empty-note", text: "Tidak tersedia." }),

      h("div", { class: "eyebrow", style: "margin:16px 0 8px",
        text: `Seluruh kata kunci${kata.length ? ` — ${int(kata.length)}` : ""}` }),
      kata.length
        ? h("div", { class: "pills" }, kata.map((w) => h("span", { class: "pill", text: w })))
        : h("div", { class: "empty-note", text: "Daftar kata tidak berhasil dimuat dari /api/lexicon." }),

      emoji.size > 0 && h("div", { class: "eyebrow", style: "margin:16px 0 8px",
        text: `Entri emoji — ${int(emoji.size)}, hanya aktif saat inferensi` }),
      emoji.size > 0 && h("div", { class: "pills" },
        [...emoji].map((e) => h("span", { class: "pill", "data-kind": "special", text: e }))
      )
    );
  });

  return h("div", { class: "card" },
    h("div", { class: "card-head" },
      h("div", {},
        h("div", { class: "card-title", text: "Isi leksikon pelabel" }),
        h("div", { class: "card-sub", text:
          "Inilah aturan yang benar-benar menghasilkan label latih. Angka di sebelah kata adalah berapa baris yang dipicunya." })
      )
    ),
    h("div", { class: "small-multiples", style: "grid-template-columns:repeat(2,1fr)" }, kolom),
    h("div", { class: "banner", "data-level": "info", style: "margin-top:16px" },
      icon(ICON_INFO, "banner-icon"),
      h("div", {},
        h("div", { class: "banner-title", text: "Entri emoji tidak berperan saat pelabelan" }),
        h("div", { class: "banner-body", text:
          "Label dihitung atas text_norm, sedangkan _norm() sudah menghapus seluruh emoji lewat " +
          "[^a-z0-9\\s] — diperiksa ulang atas seluruh korpus berlabel: nol kecocokan. Entri emoji " +
          "baru aktif saat inferensi, lewat fusi OR pada teks mentah. Kebalikannya juga berlaku: " +
          "saat inferensi hanya entri emoji yang bisa cocok, karena jalur leksikon hanya menerima " +
          "hasil EMOJI_RX.findall()." })
      )
    )
  );
}

/* --------------------------------------------------------------------------
   Tampilan Dataset & Metode
   -------------------------------------------------------------------------- */
export function renderDataset(host, R) {
  host.innerHTML = "";
  const D = R.dataset;
  const labels = R.labels;

  /* Corong data */
  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "card-head" },
        h("div", {},
          h("div", { class: "card-title", text: "Dari korpus mentah ke data latih" }),
          h("div", { class: "card-sub", text: `Sumber: ${D.source_file}` })
        )
      ),
      h("div", { class: "grid-4" },
        tile("Baris setelah preprocessing", int(D.rows_after_prep)),
        tile("Baris berlabel (≥1 label)", int(D.rows_labeled),
          `${pct(D.rows_labeled / D.rows_after_prep)} dari korpus`),
        tile("Mengandung emoji", pctRaw(D.emoji_rate_pct), "diukur pada korpus penuh"),
        tile("Panjang teks (p50 / p95)", `${D.text_len_p50} / ${D.text_len_p95}`, "kata per tweet")
      ),
      h("div", { class: "banner", "data-level": "info", style: "margin-top:16px" },
        icon(ICON_INFO, "banner-icon"),
        h("div", {},
          h("div", { class: "banner-title", text:
            `${pct(1 - D.rows_labeled / D.rows_after_prep)} korpus tidak terpakai` }),
          h("div", { class: "banner-body", text:
            `${int(D.rows_after_prep - D.rows_labeled)} baris dibuang karena tidak memuat satu pun kata kunci leksikon — pelabelan memakai pencocokan kata, bukan skor model (lihat "Asal-usul label" di bawah). Data latih akhir hanya ${int(D.rows_labeled)} baris — condong ke teks yang emosinya eksplisit secara leksikal.` })
        )
      )
    )
  );

  /* Distribusi label + kardinalitas */
  const distData = labels.map((c) => ({
    key: c,
    label: c,
    value: D.label_counts[c],
    color: LABEL_COLOR[c],
  }));
  const maxCount = Math.max(...distData.map((d) => d.value));

  const distChart = hbar({
    data: distData,
    max: maxCount,
    ticks: [0, maxCount / 2, maxCount],
    format: (v) => int(v),
    tickFormat: (v) => int(Math.round(v)),
    gutterRight: 62,
    tooltipHtml: (d) =>
      `<div class="tt-title">${d.label}</div>` +
      `<div class="tt-row">${int(d.value)} baris (${pct(d.value / D.rows_labeled)})</div>`,
  });

  const distTable = table(
    [
      { label: "Emosi", get: (d) => esc(d.label) },
      { label: "Jumlah", get: (d) => int(d.value) },
      { label: "Porsi", get: (d) => pct(d.value / D.rows_labeled) },
    ],
    distData
  );
  const dPair = withTableView(distChart, distTable);

  const comboRows = D.top_combos.map((c) => ({
    combo: c.combo.map((b, i) => (b ? labels[i] : null)).filter(Boolean).join(" + ") || "(kosong)",
    count: c.count,
  }));

  host.appendChild(
    h("div", { class: "grid-2" },
      h("div", { class: "card" },
        h("div", { class: "chart-head" },
          h("div", {},
            h("div", { class: "card-title", text: "Distribusi label" }),
            h("div", { class: "card-sub", text: "Satu baris bisa punya lebih dari satu label" })
          ),
          h("div", { class: "spacer" }),
          dPair._toggle
        ),
        dPair,
        h("div", { class: "banner", "data-level": "warning", style: "margin-top:16px" },
          icon(ICON_ALERT, "banner-icon"),
          h("div", {},
            h("div", { class: "banner-title", text: "Ketimpangan kelas" }),
            h("div", { class: "banner-body", text: `Baris berlabel senang ${times(D.label_counts.senang / D.label_counts.marah)} lipat jumlah baris berlabel marah. Inilah sebab F1 label marah jatuh pada beberapa model pembanding.` })
          )
        )
      ),
      h("div", { class: "card" },
        h("div", { class: "card-head" },
          h("div", {},
            h("div", { class: "card-title", text: "Kardinalitas label" }),
            h("div", { class: "card-sub", text: "Berapa banyak label per tweet" })
          )
        ),
        h("div", { class: "grid-2" },
          tile("Rata-rata label/tweet", dec(D.cardinality_mean, 2)),
          tile("Maksimum", String(D.cardinality_max))
        ),
        h("div", { class: "eyebrow", style: "margin:20px 0 10px", text: "Kombinasi terbanyak" }),
        table(
          [
            { label: "Kombinasi", get: (r) => esc(r.combo) },
            { label: "Jumlah", get: (r) => int(r.count) },
            { label: "Porsi", get: (r) => pct(r.count / D.rows_labeled) },
          ],
          comboRows
        )
      )
    )
  );

  /* Split + hyperparameter */
  host.appendChild(
    h("div", { class: "grid-2" },
      h("div", { class: "card" },
        h("div", { class: "card-head" },
          h("div", {},
            h("div", { class: "card-title", text: "Pembagian data" }),
            h("div", { class: "card-sub", text: "train_test_split 80/10/10, random_state=42, tanpa stratify" })
          )
        ),
        h("div", { class: "grid-3" },
          tile("Train", int(D.split.train)),
          tile("Validation", int(D.split.val)),
          tile("Test", int(D.split.test))
        )
      ),
      h("div", { class: "card" },
        h("div", { class: "card-head" },
          h("div", {}, h("div", { class: "card-title", text: "Konfigurasi training" }))
        ),
        table(
          [
            { label: "Parameter", get: (r) => esc(r[0]) },
            { label: "Nilai", get: (r) => esc(r[1]) },
          ],
          [
            ["max_len", String(R.training.max_len)],
            ["batch_size", String(R.training.batch_size)],
            ["epochs", String(R.training.epochs)],
            ["learning_rate", String(R.training.learning_rate)],
            ["optimizer", R.training.optimizer],
            ["scheduler", R.training.scheduler],
            ["loss", R.training.loss],
            ["seed", String(R.training.seed)],
            ["threshold", R.training.threshold_search],
            ["device", `${R.training.device} · torch ${R.training.torch}`],
          ]
        )
      )
    )
  );

  /* Pelabelan otomatis */
  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "card-head" },
        h("div", {},
          h("div", { class: "card-title", text: "Asal-usul label" }),
          h("div", { class: "card-sub", text: "Bagian metodologi yang paling perlu dijelaskan saat sidang" })
        )
      ),
      h("div", { class: "grid-2" },
        h("div", {},
          h("div", { class: "eyebrow", style: "margin-bottom:8px", text: "Model pelabel" }),
          h("div", { class: "mono", style: "font-size:var(--text-sm);word-break:break-all", text: D.labeler_model }),
          h("div", { class: "secondary", style: "margin-top:10px;font-size:var(--text-sm)", text: `Strategi: ${D.labeler_strategy}` })
        ),
        h("div", {},
          h("div", { class: "eyebrow", style: "margin-bottom:8px", text: "Implikasinya" }),
          h("div", { class: "secondary", style: "font-size:var(--text-sm)", text: D.label_provenance })
        )
      )
    )
  );

  /* Isi leksikon — aturan yang sebenarnya menghasilkan label */
  host.appendChild(renderLexicon(R));

  /* Catatan metodologis */
  host.appendChild(
    h("div", { class: "card" },
      h("div", { class: "card-head" },
        h("div", {},
          h("div", { class: "card-title", text: "Catatan metodologis" }),
          h("div", { class: "card-sub", text: "Batasan yang melekat pada angka di halaman Evaluasi" })
        )
      ),
      h("div", { class: "stack", style: "gap:12px" },
        R.caveats.map((c) =>
          h("div", { class: "caveat", "data-level": c.level },
            icon(ICON_ALERT, "caveat-mark"),
            h("div", {},
              h("div", { class: "caveat-title", text: c.title }),
              h("div", { class: "caveat-body", text: c.body })
            )
          )
        )
      )
    )
  );
}
