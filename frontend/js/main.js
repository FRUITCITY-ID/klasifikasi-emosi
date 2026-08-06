/* ==========================================================================
   SIPEMO — bootstrap aplikasi
   ========================================================================== */

import { api, apiBase, isRemote } from "./api.js";
import {
  h,
  LABEL_COLOR,
  renderDataset,
  renderEvaluation,
  renderModelBanner,
  renderResult,
  renderStatusChip,
} from "./views.js";
import { hideTip } from "./charts.js";

const $ = (sel) => document.querySelector(sel);

const SAMPLES = [
  { emo: "senang", tag: "Wisuda", text: "Akhirnya wisuda juga! Bahagia banget 🥳 perjuangan selama ini terbayar" },
  { emo: "sedih", tag: "Kehilangan", text: "Sedih banget kehilangan sahabat sejak kecil 😭 rasanya hampa" },
  { emo: "marah", tag: "Pelayanan", text: "Kesel bgt sm pelayanan yg lambat ini 😡 bikin muak" },
  { emo: "takut", tag: "Sidang", text: "Deg-degan mau sidang besok, takut gak bisa jawab penguji 😨" },
  { emo: "sedih", tag: "Dua emosi", text: "Seneng akhirnya lulus tp sedih hrs pindah kota 😊😢 campur aduk rasanya" },
  { emo: "senang", tag: "Uji slang + URL", text: "@budi cek https://contoh.co.id ya #SkripsiLife gw udh seneng bgt sm hasilnya" },
  /* Contoh batas sistem, sengaja disediakan. Kalimat ini tidak memuat emosi
     apa pun, tetapi model tetap mengeluarkan label — biasanya senang. Lebih
     baik keterbatasan itu ditunjukkan sendiri daripada ditemukan penguji. */
  { emo: null, tag: "Batas: kalimat netral", text: "Rapat dimulai pukul sembilan pagi di ruang A" },
];

const state = {
  health: null,
  research: null,
  busy: false,
};

/* --------------------------------------------------------------------------
   Tema
   -------------------------------------------------------------------------- */
function initTheme() {
  const saved = localStorage.getItem("sipemo-theme");
  if (saved === "light" || saved === "dark") {
    document.documentElement.dataset.theme = saved;
  }
  $("#themeToggle").addEventListener("click", () => {
    const current =
      document.documentElement.dataset.theme ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("sipemo-theme", next);
  });
}

/* --------------------------------------------------------------------------
   Tab
   -------------------------------------------------------------------------- */
function initTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.view));
    tab.addEventListener("keydown", (e) => {
      const i = tabs.indexOf(tab);
      let next = null;
      if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
      if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
      if (next) {
        e.preventDefault();
        next.focus();
        selectTab(next.dataset.view);
      }
    });
  });
}

function selectTab(name) {
  hideTip();
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.view === name;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.hidden = v.dataset.view !== name;
  });
}

/* --------------------------------------------------------------------------
   Contoh teks
   -------------------------------------------------------------------------- */
function initSamples() {
  const host = $("#samples");
  SAMPLES.forEach((s) => {
    host.appendChild(
      h("button", {
          class: "sample",
          type: "button",
          title: s.text,
          onClick: () => {
            $("#input").value = s.text;
            analyze();
          },
        },
        h("span", {
          class: "swatch",
          /* s.emo null = contoh netral: tidak ada emosi yang boleh diwakili,
             jadi swatch-nya abu-abu, bukan warna label mana pun. */
          style: `background:${s.emo ? LABEL_COLOR[s.emo] : "var(--text-muted)"}`,
        }),
        s.tag
      )
    );
  });
}

/* --------------------------------------------------------------------------
   Analisis
   -------------------------------------------------------------------------- */
async function analyze() {
  const text = $("#input").value.trim();
  const results = $("#results");

  if (!text) {
    $("#input").focus();
    return;
  }
  if (state.busy) return;
  if (!state.health?.model?.loaded) {
    document.querySelector("#modelBanner .banner")?.scrollIntoView({ block: "center" });
    return;
  }

  state.busy = true;
  const btn = $("#runBtn");
  btn.disabled = true;
  const originalLabel = btn.innerHTML;
  btn.innerHTML = '<span class="spinner"></span> Menganalisis…';
  /* Tahan render sebelumnya dengan opasitas rendah — tanpa lompatan layout */
  if (results.childElementCount) results.classList.add("is-stale");

  try {
    const r = await api.predict(text);
    results.classList.remove("is-stale");
    renderResult(results, r);
    results.hidden = false;
  } catch (e) {
    results.classList.remove("is-stale");
    results.innerHTML = "";
    results.hidden = false;
    results.appendChild(
      h("div", { class: "banner", "data-level": "error" },
        h("div", {},
          h("div", { class: "banner-title", text: "Analisis gagal" }),
          h("div", { class: "banner-body", text: e.message })
        )
      )
    );
  } finally {
    state.busy = false;
    btn.disabled = false;
    btn.innerHTML = originalLabel;
  }
}

/* --------------------------------------------------------------------------
   Muat status model
   -------------------------------------------------------------------------- */
async function loadHealth() {
  try {
    state.health = await api.health();
  } catch (e) {
    /* Penyebab kegagalan berbeda tergantung di mana antarmuka ini dibuka, dan
       saran yang salah lebih buruk daripada tidak ada saran. Versi lokal
       biasanya gagal karena uvicorn belum jalan; versi Pages hampir selalu
       karena kontainer Railway sedang bangun dari tidur — model IndoBERT butuh
       waktu untuk dimuat, dan permintaan pertama bisa habis waktu. */
    state.health = {
      model: {
        status: "error",
        loaded: false,
        message: isRemote
          ? `Tidak bisa menghubungi backend di ${apiBase} — ${e.message}. ` +
            "Server bisa jadi sedang bangun dari mode tidur; pemuatan model " +
            "IndoBERT memakan waktu sekitar satu menit pada permintaan " +
            "pertama. Tunggu sebentar lalu klik 'Muat ulang model'."
          : `Tidak bisa menghubungi server: ${e.message}. Pastikan uvicorn berjalan.`,
        checkpoint_path: "—",
        warnings: [],
      },
    };
  }

  renderStatusChip($("#statusChip"), state.health);
  renderModelBanner($("#modelBanner"), state.health);

  const ready = !!state.health.model?.loaded;
  $("#runBtn").disabled = !ready;
  $("#input").disabled = !ready;

  const retry = $("#retryLoad");
  if (retry) {
    retry.addEventListener("click", async () => {
      retry.disabled = true;
      retry.textContent = "Memuat…";
      try {
        await api.reload();
      } catch { /* status terbaru diambil ulang di bawah */ }
      await loadHealth();
    });
  }
}

async function loadResearch() {
  try {
    state.research = await api.research();

    /* Leksikon dimuat terpisah dan BOLEH gagal: kalau /api/lexicon bermasalah,
       kartu "Isi leksikon pelabel" cukup menampilkan pemicu teratas dari
       /api/research. Satu endpoint tambahan tidak boleh menjatuhkan seluruh
       tab Dataset. */
    try {
      state.research.lexicon = await api.lexicon();
    } catch {
      state.research.lexicon = null;
    }

    renderEvaluation($("#evalView"), state.research);
    renderDataset($("#dataView"), state.research);
  } catch (e) {
    const msg = h("div", { class: "banner", "data-level": "error" },
      h("div", {},
        h("div", { class: "banner-title", text: "Gagal memuat data penelitian" }),
        h("div", { class: "banner-body", text: e.message })
      )
    );
    $("#evalView").appendChild(msg.cloneNode(true));
    $("#dataView").appendChild(msg);
  }
}

/* --------------------------------------------------------------------------
   Start
   -------------------------------------------------------------------------- */
function init() {
  initTheme();
  initTabs();
  initSamples();

  $("#runBtn").addEventListener("click", analyze);
  $("#clearBtn").addEventListener("click", () => {
    $("#input").value = "";
    $("#results").hidden = true;
    $("#results").innerHTML = "";
    $("#input").focus();
  });
  $("#input").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
  });
  window.addEventListener("scroll", hideTip, { passive: true });

  loadHealth();
  loadResearch();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
