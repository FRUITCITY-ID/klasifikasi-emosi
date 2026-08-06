/* ==========================================================================
   SIPEMO — pemformatan angka (locale id-ID)

   Kenapa modul ini ada
   --------------------
   Versi sebelumnya mencampur dua konvensi pada halaman yang sama: jumlah baris
   lewat toLocaleString("id-ID") sehingga titik berarti PEMISAH RIBUAN, tetapi
   persentase dan metrik lewat toFixed() sehingga titik berarti KOMA DESIMAL.
   Hasilnya bentuk seperti "1.234 baris" dan "56.7%" tampil berdampingan —
   karakter yang sama dengan dua arti berbeda, dan yang kedua bisa terbaca
   sebagai lima puluh enam ribu tujuh oleh pembaca Indonesia.

   Contoh di bawah sengaja memakai angka karangan, bukan angka penelitian:
   verify_research.py menolak angka penelitian apa pun di dalam frontend/js/,
   termasuk yang hanya muncul di komentar, supaya tidak ada salinan yang bisa
   diam-diam jadi basi.

   Seluruh angka yang TAMPIL sebagai data sekarang melewati modul ini: ribuan
   dengan titik, desimal dengan koma. Satu pengecualian yang disengaja — blok
   .formula menampilkan ekspresi Python apa adanya, dan di sana 0.30 memang
   harus tertulis 0.30 karena itu literal kode, bukan angka bacaan.
   ========================================================================== */

const nf = (min, max) =>
  new Intl.NumberFormat("id-ID", {
    minimumFractionDigits: min,
    maximumFractionDigits: max,
  });

const CACHE = new Map();
function fmt(digits) {
  if (!CACHE.has(digits)) CACHE.set(digits, nf(digits, digits));
  return CACHE.get(digits);
}

/** Bilangan bulat dengan pemisah ribuan: 1234567 → "1.234.567" */
export const int = (v) => fmt(0).format(v);

/** Desimal dengan koma, jumlah digit tetap: (0.42185, 3) → "0,422" */
export const dec = (v, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : fmt(digits).format(v);

/** Pecahan 0..1 menjadi persen: 0.075 → "7,5%" */
export const pct = (v, digits = 1) => `${fmt(digits).format(v * 100)}%`;

/** Angka yang sudah dalam satuan persen: 12.5 → "12,5%" */
export const pctRaw = (v, digits = 1) => `${fmt(digits).format(v)}%`;

/** Faktor perbandingan: 3.42 → "3,4×" */
export const times = (v, digits = 1) => `${fmt(digits).format(v)}×`;
