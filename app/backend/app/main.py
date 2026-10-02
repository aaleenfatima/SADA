"""
SADA inference API.

Endpoints:
    POST /analyze            -> {job_id, cached?}  (returns immediately)
    GET  /jobs/{job_id}      -> status + staged progress + result when done
    GET  /featured           -> precomputed demo targets (instant)
    GET  /healthz            -> model artifact status

Nothing blocks: a cold MAST fetch takes 20-90s, so /analyze queues work and the
client polls /jobs/{id}. Cached stars skip the queue entirely.
"""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

# Load app/backend/.env BEFORE importing sibling modules — inference.py reads
# SADA_MODELS_DIR at import time, so if dotenv ran after that import, the
# env var would already be missing and this fix would silently do nothing.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import agent, inference, llm, store

app = FastAPI(title="SADA Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SADA_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CATALOG_PATH = os.environ.get("SADA_CATALOG_PATH", "data/catalog_lookup.json")
_catalog: dict = {}


class AnalyzeRequest(BaseModel):
    star_id: int = Field(..., description="Kepler Input Catalog ID, e.g. 11442793")
    force_live_search: bool = Field(
        False,
        description="Ignore DR25 catalog ephemerides and run a live BLS search "
                     "even for known targets (slower; demonstrates the search path).")
    use_cache: bool = True


@app.on_event("startup")
def startup():
    store.init_db()
    inference.load_models()
    global _catalog
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            _catalog = json.load(f)


@app.get("/healthz")
def healthz():
    status = inference.models_status()
    ready = status["cnn_loaded"]
    return {
        "ready": ready,
        "models": status,
        "catalog_entries": len(_catalog),
        "note": None if ready else
            "CNN artifact missing — export it from Colab "
            "(scripts/export_models_colab.py). The API will not fabricate "
            "predictions without a trained model.",
    }


@app.get("/featured")
def featured():
    return {"stars": store.list_featured()}


@app.post("/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    if not inference.models_status()["cnn_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="No trained CNN loaded; refusing to return predictions. "
                   "See /healthz.")

    if req.use_cache and not req.force_live_search:
        cached = store.get_cached(req.star_id)
        if cached is not None:
            return {"cached": True, "job_id": None, "result": cached}

    job_id = store.create_job(req.star_id)
    background_tasks.add_task(_run_job, job_id, req.star_id, req.force_live_search)
    return {"cached": False, "job_id": job_id, "result": None}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job")
    return job


def _run_job(job_id: str, star_id: int, force_live_search: bool):
    def progress(stage: str, message: str):
        store.append_progress(job_id, stage, message)

    try:
        catalog_params = None if force_live_search else _catalog.get(str(star_id))
        result = agent.analyze_star(star_id, catalog_params=catalog_params,
                                     progress=progress)

        if result.get("status") == "complete":
            progress("report", "Writing vetting report...")
            for cand in result["candidates"]:
                if cand.get("status") == "ok":
                    cand["report"] = llm.generate_report(
                        cand, star_id,
                        rf_unavailable_reason=result.get("rf_unavailable_reason"))
            store.put_cached(star_id, result)

        store.finish_job(job_id, result)
    except Exception as e:  # noqa: BLE001 - surfaced to the client as job failure
        store.fail_job(job_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")