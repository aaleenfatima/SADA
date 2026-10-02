# SADA — Inference API (Layer 2)

FastAPI backend that turns the Layer 1 research models into a working vetting
agent. Given only a star ID, it fetches the lightcurve from NASA MAST, searches
for candidate periods, classifies each one, and decides whether to accept,
reject, or escalate to human review.

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then `GET /healthz`. It will report `ready: false` until model artifacts exist —
see "Model artifacts" below.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/analyze` | Queue a star. Returns `{job_id}` immediately, or a cached result. |
| `GET` | `/jobs/{job_id}` | Poll status + staged progress; result when complete. |
| `GET` | `/featured` | Precomputed demo stars, instant. |
| `GET` | `/healthz` | Model artifact status. |

`POST /analyze` body:

```json
{ "star_id": 11442793, "force_live_search": false, "use_cache": true }
```

## Why a job queue instead of a plain endpoint

A cold MAST fetch takes 20–90s and BLS adds seconds more; free-tier hosts cut
HTTP requests at ~30s. So nothing blocks the request thread — the client polls
and renders staged progress:

```
✓ Fetched lightcurve (14 quarters)
✓ BLS found 3 candidate period(s): 3.5217d, 7.0433d, 14.0871d
⟳ Folding + classifying candidate 2 of 3...
```

Cached stars skip the queue and return in <1s.

## The agent loop

1. Fetch lightcurve (MAST, or cache)
2. Determine ephemerides — catalog if the star is a known DR25 TCE, otherwise a
   live BLS period search
3. Classify **each** candidate period independently
4. If a verdict is ambiguous *and* the period came from BLS, refine the grid and
   re-check — keeping the refinement only if it sharpened the verdict
5. Run an explicit odd/even transit-depth test for eclipsing binaries
6. Decide per candidate: accept / reject / escalate

One star returns **one result per candidate**, not one verdict. 35% of TCEs in
our dataset share a host star, so a single answer per star would be wrong.

### What triggers escalation, and why

Each rule comes from a measured result in the Layer 1 research, not from taste:

| Trigger | Measured basis |
|---|---|
| Confidence in 0.40–0.80 band | CNN Brier = 0.139 vs RF's 0.083; the reliability diagram shows the CNN overconfident precisely in this range |
| Odd/even depths differ >3σ with a positive verdict | CNN misclassifies likely eclipsing binaries at 23.6% vs RF's 10.9% |
| CNN and RF disagree | Both available only on catalog targets; disagreement is a human's call |
| Positive verdict at a low-order period ratio of a stronger candidate | May be a BLS alias *or* a genuine resonant companion — the periodogram can't distinguish them |

## Two design decisions worth knowing about

**1. RF is not served on live-search targets.** RF_main was trained on catalog
`tce_period` (days), `tce_duration` (hours), `tce_depth` (ppm) and
`tce_model_snr` (a pipeline-specific statistic). A live BLS search produces its
own period/duration/depth in different units and scales, and no SNR equivalent
at all. Feeding BLS values into RF would be a silent distribution shift making
its output mean something different from what the training metrics claim. So RF
runs only where real catalog parameters exist; the response says
`rf_available: false` with a reason rather than hiding it.

**2. Period aliases are flagged, not deleted.** BLS reports aliases at integer
and rational multiples of the true period. Discarding every small-integer ratio
would be the obvious fix — but real multi-planet systems are frequently in 2:1,
3:2 or 4:3 resonance (Kepler-90 among them), and a period ratio alone cannot
distinguish a resonant companion from an alias. Deleting them would mean this
tool could never find a resonant system. They are annotated
(`possible_alias_of`) and escalated instead.

## Preprocessing parity

`app/preprocessing.py` reproduces the training pipeline exactly: sigma-clip at
3σ → fold on period/epoch → bin to 200 → reject on any NaN → z-score → reject
if length ≠ 200. **Changing it silently invalidates the served model.** The
source of truth is `research/notebooks/02_eda.ipynb`.

## Repo layout

This lives at `app/backend/` in the main SADA repo. Models stay in
`research/models/` (single source of truth — don't duplicate weights into
`app/backend`, or a retrain-without-recopy will silently serve stale weights):

```bash
# app/backend/.env
SADA_MODELS_DIR=../../research/models
SADA_CATALOG_PATH=data/catalog_lookup.json
```

## Model artifacts

Not fully present yet:

| Artifact | Status |
|---|---|
| `research/models/cnn_1d_best.pt` | Present. Loader accepts this filename or `cnn_1d.pt`, and unwraps either a raw `state_dict` or a wrapped checkpoint (`model_state_dict`/`state_dict`/`model` keys) — verify at `/healthz` which shape yours turned out to be. |
| `research/models/rf_main.joblib` | **Missing.** Export from Colab — see below. Until then, `rf_available` is always `false`, even for catalog targets. |
| `data/catalog_lookup.json` | Build locally, no Colab needed: |

```bash
python scripts/build_catalog_lookup.py \
    --csv ../../research/data/balanced_tce.csv \
    --out data/catalog_lookup.json
```

This only covers the 8,074-TCE balanced set (6,443 stars) — a star outside it
falls back to live BLS search automatically.

For the CNN/RF artifacts specifically:

```bash
python scripts/export_models_colab.py   # prints the snippet to paste
```

The export asserts the CNN has exactly 3,953 parameters and RF's feature order
matches, so a wrong-architecture or wrong-column-order artifact fails loudly at
export rather than silently serving wrong predictions.

Without a CNN artifact, `/analyze` returns **503**. It does not fall back to
fabricated output.

## Precaching the demo

```bash
python -m scripts.precache
```

Computes the featured stars and writes them to `cache/sada.db`. Commit that file
so the demo works instantly on a cold start, and even if MAST is unreachable.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SADA_MODELS_DIR` | `models` | Where to load artifacts from |
| `SADA_DB_PATH` | `cache/sada.db` | Job + cache store |
| `SADA_CATALOG_PATH` | `data/catalog_lookup.json` | DR25 lookup |
| `ANTHROPIC_API_KEY` | *(unset)* | Enables LLM reports; without it a deterministic template is used |
| `SADA_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## Tests

```bash
python test_smoke.py
```

Exercises preprocessing parity, BLS period recovery on an injected transit,
alias handling, the full agent loop on both paths, report generation, and the
API endpoints — using a synthetic lightcurve and a stubbed model, so it needs no
MAST access, no network, and no trained weights.
