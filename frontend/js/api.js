/* ==========================================================================
   SIPEMO — klien API
   ========================================================================== */

/* Alamat backend saat antarmuka ini TIDAK disajikan oleh backend-nya sendiri.
   Diisi saat deployment; lihat resolveBase() di bawah. */
const REMOTE_API = "https://sipemo-api-production.up.railway.app";

/* Backend bisa berada di dua tempat, dan keduanya harus tetap berfungsi:

   - Lokal (jalankan-sipemo.bat): uvicorn menyajikan index.html sekaligus
     /api/*, jadi semuanya satu origin dan BASE cukup string kosong. Memakai
     URL absolut di sini justru merugikan — permintaan akan keluar ke internet
     padahal modelnya berjalan di laptop yang sama.

   - GitHub Pages: halaman statis tanpa Python sama sekali, sehingga /api/*
     harus ditujukan ke Railway.

   Pembedanya hostname, bukan flag build, supaya satu berkas yang sama bisa
   dipakai di kedua tempat tanpa langkah build. */
function resolveBase() {
  /* Jalan keluar untuk pengujian: arahkan halaman Pages ke backend lokal, atau
     sebaliknya, lewat konsol —
       localStorage.setItem("sipemo-api", "http://127.0.0.1:8000")
     Kosongkan dengan removeItem untuk kembali ke perilaku normal. */
  const override = localStorage.getItem("sipemo-api");
  if (override) return override.replace(/\/+$/, "");

  const host = location.hostname;
  if (host === "localhost" || host === "127.0.0.1" || host === "[::1]") return "";

  return REMOTE_API.replace(/\/+$/, "");
}

const BASE = resolveBase();

/* Diekspor supaya lapisan tampilan bisa menyebut alamat sebenarnya saat
   koneksi gagal — "tidak bisa menghubungi server" jauh lebih berguna kalau
   disertai server mana yang dicoba. */
export const apiBase = BASE;
export const isRemote = BASE !== "";

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let payload = null;
  try {
    payload = await res.json();
  } catch {
    payload = null;
  }

  if (!res.ok) {
    const detail = payload && payload.detail;
    const err = new Error(
      typeof detail === "string"
        ? detail
        : (detail && detail.message) || `HTTP ${res.status}`
    );
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return payload;
}

export const api = {
  health: () => request("/api/health"),
  research: () => request("/api/research"),
  lexicon: () => request("/api/lexicon"),
  reload: () => request("/api/reload", { method: "POST" }),
  predict: (text) =>
    request("/api/predict", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
};
