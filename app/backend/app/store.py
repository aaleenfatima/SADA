"""
Job store + result cache (SQLite).

Why SQLite rather than an in-memory dict: free-tier dynos restart and scale to
zero, and an in-memory cache dies with them. A small SQLite file on a mounted
volume (or committed alongside precomputed demo stars) keeps the featured
targets instant even after a cold start.

Why a job queue at all: a cold MAST fetch takes 20-90s and BLS adds seconds
more, while free-tier platforms cut HTTP requests at ~30s. Nothing here blocks
the request thread; the client polls.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Optional

DB_PATH = os.environ.get("SADA_DB_PATH", "cache/sada.db")

_lock = threading.Lock()


def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock, _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                star_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                stage TEXT,
                progress_json TEXT NOT NULL DEFAULT '[]',
                result_json TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results_cache (
                star_id INTEGER PRIMARY KEY,
                result_json TEXT NOT NULL,
                is_featured INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )""")
        conn.commit()


# --- cache ------------------------------------------------------------------

def get_cached(star_id: int) -> Optional[dict]:
    with _lock, _conn() as conn:
        row = conn.execute("SELECT result_json FROM results_cache WHERE star_id = ?",
                            (star_id,)).fetchone()
    return json.loads(row["result_json"]) if row else None


def put_cached(star_id: int, result: dict, is_featured: bool = False):
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO results_cache (star_id, result_json, is_featured, created_at)"
            " VALUES (?, ?, ?, ?)",
            (star_id, json.dumps(result), 1 if is_featured else 0, time.time()))
        conn.commit()


def list_featured() -> list:
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT star_id, result_json FROM results_cache WHERE is_featured = 1"
            " ORDER BY star_id").fetchall()
    return [{"star_id": r["star_id"], **json.loads(r["result_json"])} for r in rows]


# --- jobs -------------------------------------------------------------------

def create_job(star_id: int) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, star_id, status, stage, progress_json,"
            " created_at, updated_at) VALUES (?, ?, 'queued', NULL, '[]', ?, ?)",
            (job_id, star_id, now, now))
        conn.commit()
    return job_id


def append_progress(job_id: str, stage: str, message: str):
    with _lock, _conn() as conn:
        row = conn.execute("SELECT progress_json FROM jobs WHERE job_id = ?",
                            (job_id,)).fetchone()
        if row is None:
            return
        entries = json.loads(row["progress_json"])
        entries.append({"stage": stage, "message": message, "at": time.time()})
        conn.execute(
            "UPDATE jobs SET progress_json = ?, stage = ?, status = 'running',"
            " updated_at = ? WHERE job_id = ?",
            (json.dumps(entries), stage, time.time(), job_id))
        conn.commit()


def finish_job(job_id: str, result: dict):
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'complete', result_json = ?, updated_at = ?"
            " WHERE job_id = ?", (json.dumps(result), time.time(), job_id))
        conn.commit()


def fail_job(job_id: str, error: str):
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, updated_at = ?"
            " WHERE job_id = ?", (error, time.time(), job_id))
        conn.commit()


def get_job(job_id: str) -> Optional[dict]:
    with _lock, _conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return {
        "job_id": row["job_id"],
        "star_id": row["star_id"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": json.loads(row["progress_json"]),
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
