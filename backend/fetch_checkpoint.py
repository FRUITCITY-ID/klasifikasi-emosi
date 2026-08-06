"""
SIPEMO — pengunduh checkpoint untuk deployment.

Checkpoint `best_indobert_base.pt` berukuran ~475 MiB. GitHub menolak file di
atas 100 MiB di dalam riwayat git, jadi file itu TIDAK ikut di-commit — ia
dilampirkan sebagai aset GitHub Release, yang batasnya 2 GiB dan tidak memakan
kuota Git LFS.

Skrip ini menjembatani keduanya: saat image Docker dibangun, checkpoint diunduh
dari URL rilis lalu ditaruh di tempat yang dicari `config.CHECKPOINT_PATH`.
Dijalankan saat BUILD, bukan saat start, supaya cold start Railway tidak
membayar unduhan 475 MiB setiap kali kontainer bangun.

Pemakaian:
    python backend/fetch_checkpoint.py

Aman dipanggil berulang: kalau file sudah ada dan sha256-nya cocok, unduhan
dilewati.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import config

# Sidik jari checkpoint IndoBERT-base yang dilayani. Dihitung dari file yang
# sama yang menghasilkan angka di backend/research.py, jadi kalau aset rilis
# suatu saat diganti tanpa sengaja, verifikasi ini akan menangkapnya sebelum
# model dimuat dan menghasilkan prediksi yang tidak sesuai metrik yang ditampilkan.
EXPECTED_SHA256 = os.environ.get("SIPEMO_CHECKPOINT_SHA256", "").strip().lower()

DEFAULT_URL = (
    "https://github.com/FRUITCITY-ID/klasifikasi-emosi/releases/download/"
    "v1.0-model/best_indobert_base.pt"
)

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def already_good(path: Path) -> bool:
    """True kalau file di disk sudah checkpoint yang benar."""
    if not path.exists():
        return False
    if not EXPECTED_SHA256:
        # Tanpa sidik jari, satu-satunya yang bisa dipastikan adalah file itu
        # bukan sisa unduhan yang terpotong. Checkpoint asli ~475 MiB.
        return path.stat().st_size > 400 * 1024 * 1024
    return sha256_of(path) == EXPECTED_SHA256


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    print(f"[fetch] mengunduh {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "sipemo-deploy"})
    with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        next_mark = 10
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct >= next_mark:
                    print(f"[fetch] {pct}% ({done >> 20} MiB / {total >> 20} MiB)")
                    next_mark = pct - pct % 10 + 10

    if EXPECTED_SHA256:
        got = sha256_of(tmp)
        if got != EXPECTED_SHA256:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"[fetch] GAGAL: sha256 tidak cocok.\n"
                f"  diharapkan {EXPECTED_SHA256}\n"
                f"  didapat    {got}\n"
                "Aset rilis kemungkinan bukan checkpoint yang dipakai untuk "
                "menghitung metrik di backend/research.py. Build dihentikan "
                "daripada menyajikan bobot yang salah."
            )
        print("[fetch] sha256 cocok.")

    # Rename baru dilakukan setelah verifikasi lulus, supaya file di path final
    # tidak pernah dalam keadaan setengah jadi atau salah isi.
    tmp.replace(dest)
    print(f"[fetch] selesai: {dest} ({dest.stat().st_size >> 20} MiB)")


def main() -> int:
    url = os.environ.get("SIPEMO_CHECKPOINT_URL", DEFAULT_URL).strip()
    dest = config.CHECKPOINT_PATH

    if already_good(dest):
        print(f"[fetch] checkpoint sudah ada di {dest} — unduhan dilewati.")
        return 0

    if not url:
        print(
            "[fetch] SIPEMO_CHECKPOINT_URL kosong dan checkpoint tidak ada. "
            "Server tetap akan start, tetapi /api/health melaporkan "
            "'no_checkpoint' dan prediksi dinonaktifkan.",
            file=sys.stderr,
        )
        return 0

    try:
        download(url, dest)
    except urllib.error.HTTPError as e:
        print(f"[fetch] HTTP {e.code} saat mengambil {url}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"[fetch] gagal menghubungi {url}: {e.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
