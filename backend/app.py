"""
SIPEMO — API server.

Jalankan dari folder backend/:
    uvicorn app:app --reload --port 8000

Lalu buka http://127.0.0.1:8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import model as model_mod
import research
from preprocessing import LEX, SLANG


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class BatchRequest(BaseModel):
    # Batasan per ITEM harus sama dengan PredictRequest. Sebelumnya min_length
    # dan max_length hanya membatasi panjang daftarnya, sehingga
    # {"texts": [""]} lolos padahal {"text": ""} ditolak 422 — dua endpoint
    # dengan aturan berbeda untuk masukan yang sama.
    texts: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(
        ..., min_length=1, max_length=50
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[sipemo] memuat model…")
    state = model_mod.load()
    print(f"[sipemo] status={state.status} — {state.message}")
    for w in state.warnings:
        print(f"[sipemo] WARNING: {w}")
    yield


app = FastAPI(
    title="SIPEMO API",
    description="Klasifikasi emosi multi-label teks Indonesia (IndoBERT-base).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Status model — UI memakai ini untuk memutuskan apa yang boleh ditampilkan."""
    return {
        "model": model_mod.STATE.as_dict(),
        "labels": config.LABEL_COLS,
        "label_meta": config.LABEL_META,
        "max_len": config.MAX_LEN,
    }


@app.post("/api/reload")
def reload_model():
    """Muat ulang model — berguna setelah menaruh checkpoint baru."""
    state = model_mod.load()
    return {"model": state.as_dict()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not model_mod.STATE.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "status": model_mod.STATE.status,
                "message": model_mod.STATE.message,
            },
        )
    try:
        return model_mod.predict(req.text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


@app.post("/api/predict/batch")
def predict_batch(req: BatchRequest):
    if not model_mod.STATE.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "status": model_mod.STATE.status,
                "message": model_mod.STATE.message,
            },
        )
    return {"results": [model_mod.predict(t) for t in req.texts]}


@app.get("/api/research")
def get_research():
    """Metadata & metrik penelitian — angkanya beku dari output notebook."""
    served = research.result_for(config.SERVED_MODEL_NAME)
    return {
        "served_model": config.SERVED_MODEL_NAME,
        "served_model_id": config.SERVED_MODEL_ID,
        "served_result": served,
        "results": research.MODEL_RESULTS,
        "dataset": research.DATASET,
        "training": research.TRAINING,
        "caveats": research.CAVEATS,
        "labels": config.LABEL_COLS,
        "label_meta": config.LABEL_META,
        "notebook": "multilabel_bert_comparison.ipynb",
    }


@app.get("/api/lexicon")
def get_lexicon():
    """Isi kamus SLANG & leksikon emoji yang dipakai pipeline."""
    emoji_only = {
        label: sorted(w for w in words if not w.isascii())
        for label, words in LEX.items()
    }
    return {
        "slang": SLANG,
        "slang_size": len(SLANG),
        "lexicon": {k: sorted(v) for k, v in LEX.items()},
        "emoji_lexicon": emoji_only,
        "note": (
            "Saat inferensi hanya entri emoji yang bisa cocok, karena "
            "label_emoji_set() hanya menerima hasil EMOJI_RX.findall() dari "
            "teks mentah. Entri kata dipakai saat PELABELAN dataset: bukan "
            "sebagai fallback, melainkan sebagai satu-satunya pelabel yang "
            "benar-benar berjalan — jalur model neural di cell 3 tidak pernah "
            "menghasilkan label (lihat caveat pada /api/research). Kebalikannya "
            "juga berlaku: entri emoji tidak pernah aktif saat pelabelan, "
            "karena _norm() sudah menghapus emoji sebelum leksikon dipakai."
        ),
    }


# --------------------------------------------------------------------------
# Frontend statis
# --------------------------------------------------------------------------
if config.FRONTEND_DIR.exists():

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        """
        Wajibkan browser memvalidasi ulang aset frontend.

        StaticFiles mengirim ETag & Last-Modified tetapi tidak mengirim
        Cache-Control, sehingga browser memakai heuristik cache-nya sendiri dan
        bisa terus memakai JS lama walau file di disk sudah berubah — cukup
        menyesatkan kalau UI diperbaiki sesaat sebelum demo. 'no-cache' bukan
        berarti tanpa cache: browser tetap menyimpan salinan, hanya wajib
        menanyakan ETag dulu, jadi biasanya cuma bertukar 304 tanpa transfer.
        """
        response = await call_next(request)
        if request.url.path.startswith(("/js/", "/css/")) or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount(
        "/css", StaticFiles(directory=config.FRONTEND_DIR / "css"), name="css"
    )
    app.mount("/js", StaticFiles(directory=config.FRONTEND_DIR / "js"), name="js")

    @app.get("/")
    def index():
        return FileResponse(config.FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
