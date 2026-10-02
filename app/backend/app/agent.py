"""
The agent loop.

This is the part that makes SADA's Layer 2 agentic rather than a single
inference endpoint. Given only a star ID, the agent:

    1. fetches the lightcurve (MAST, or cache)
    2. searches for candidate periods (it does not know them in advance)
    3. classifies EACH candidate independently
    4. when a verdict is ambiguous, refines the period grid and re-checks
    5. runs an explicit odd/even depth test for eclipsing binaries
    6. decides per candidate: accept / reject / escalate to human review

Step 3 matters for a data reason: 35% of TCEs in our dataset share a host star,
so "one star -> one verdict" is wrong. A star gets one result per candidate.
"""
from __future__ import annotations

import time as _time
from typing import Callable, Optional

import numpy as np

from . import bls as bls_mod
from . import inference
from .preprocessing import RF_FEATURES, flux_shape_features, fold_bin_normalize

ProgressFn = Callable[[str, str], None]  # (stage, message) -> None


def _noop(stage: str, message: str) -> None:
    pass


def analyze_star(star_id: int,
                 catalog_params: Optional[list] = None,
                 n_candidates: int = 3,
                 progress: ProgressFn = _noop,
                 fetch_fn: Optional[Callable] = None) -> dict:
    """Run the full agent loop for one star.

    catalog_params: optional list of dicts for known DR25 TCEs on this star,
        each with tce_period / tce_time0bk / tce_duration / tce_depth /
        tce_model_snr. When present the agent uses catalog ephemerides (and can
        serve RF_main); when absent it falls back to live BLS search and serves
        the CNN only.
    fetch_fn: injectable for testing; defaults to the real MAST fetch.
    """
    started = _time.time()
    fetch_fn = fetch_fn or fetch_lightcurve

    progress("fetch", f"Fetching lightcurve for KIC {star_id} from MAST...")
    lc = fetch_fn(star_id)
    if lc is None:
        return {"star_id": star_id, "status": "failed",
                "error": "Could not retrieve a lightcurve for this target from MAST.",
                "candidates": []}

    n_quarters = getattr(lc, "n_quarters", None)
    progress("fetch", f"Retrieved lightcurve"
                       + (f" ({n_quarters} quarters)" if n_quarters else ""))

    time_arr = np.asarray(lc.time.value, dtype=float)
    raw_flux = np.asarray(lc.flux.value, dtype=float)

    # ---- Decide where ephemerides come from --------------------------------
    used_catalog = bool(catalog_params)
    if used_catalog:
        progress("search", f"Using DR25 catalog ephemerides "
                            f"({len(catalog_params)} known TCE(s) on this star)")
        candidates = [{
            "period": float(p["tce_period"]),
            "epoch": float(p["tce_time0bk"]),
            "duration": float(p.get("tce_duration", 3.0)) / 24.0,  # hours -> days
            "depth": None,
            "power": None,
            "catalog": p,
        } for p in catalog_params]
    else:
        progress("search", "No catalog parameters for this target — "
                            "running BLS period search...")
        candidates = bls_mod.search_periods(time_arr, raw_flux,
                                            n_candidates=n_candidates)
        if not candidates:
            return {"star_id": star_id, "status": "failed",
                    "error": "BLS period search found no usable candidate period "
                             "(lightcurve may be too short or too sparse).",
                    "candidates": []}
        progress("search", f"BLS found {len(candidates)} candidate period(s): "
                            + ", ".join(f"{c['period']:.4f}d" for c in candidates))

    # ---- Classify each candidate -------------------------------------------
    results = []
    for i, cand in enumerate(candidates, start=1):
        progress("classify", f"Folding + classifying candidate {i} of "
                              f"{len(candidates)} (P={cand['period']:.4f}d)...")
        result = _evaluate_candidate(lc, time_arr, raw_flux, cand,
                                      used_catalog=used_catalog, progress=progress)
        results.append(result)

    accepted = sum(1 for r in results if r["decision"] == "accept")
    escalated = sum(1 for r in results if r["decision"] == "escalate")
    rejected = sum(1 for r in results if r["decision"] == "reject")
    progress("done", f"Complete — {accepted} accepted, {escalated} flagged for "
                      f"review, {rejected} rejected")

    # rf_available means "RF will actually be attempted and could return a
    # probability" -- catalog ephemerides existing is necessary but not
    # sufficient; the RF artifact also has to be loaded. Conflating these two
    # produced a real bug: a star with real catalog data but no loaded RF
    # artifact reported rf_available=true and rf_unavailable_reason=None, so
    # downstream report generation had no correct explanation available and
    # fell back to a wrong one ("catalog parameters were not available").
    rf_loaded = inference.models_status()["rf_loaded"]
    rf_available = used_catalog and rf_loaded
    if not used_catalog:
        rf_unavailable_reason = (
            "RF_main requires DR25 catalog features (period/duration/depth/SNR) in "
            "their original units; BLS-derived values are not equivalent, so only "
            "the CNN is served for this target.")
    elif not rf_loaded:
        rf_unavailable_reason = (
            "DR25 catalog parameters were available for this target, but no "
            "RF_main model artifact is currently loaded on this server (see "
            "/healthz) — only the CNN was applied.")
    else:
        rf_unavailable_reason = None

    return {
        "star_id": star_id,
        "status": "complete",
        "ephemeris_source": "dr25_catalog" if used_catalog else "live_bls_search",
        "rf_available": rf_available,
        "rf_unavailable_reason": rf_unavailable_reason,
        "elapsed_seconds": round(_time.time() - started, 1),
        "candidates": results,
    }


def _evaluate_candidate(lc, time_arr, raw_flux, cand, used_catalog: bool,
                        progress: ProgressFn) -> dict:
    """Classify one candidate period, refining once if the verdict is ambiguous."""
    flux = fold_bin_normalize(lc, cand["period"], cand["epoch"])
    if flux is None:
        return {"period": cand["period"], "epoch": cand["epoch"],
                "status": "preprocessing_failed",
                "decision": "reject",
                "reason": "Folded lightcurve contained gaps/NaNs after binning "
                          "(same rejection rule as training).",
                "cnn_probability": None, "rf_probability": None,
                "confidence_band": None, "odd_even": None, "refined": False}

    cnn_prob = inference.predict_cnn(flux)
    refined = False

    # --- Refinement pass: only for ambiguous verdicts, and only for BLS-derived
    # periods. A catalog ephemeris is already authoritative; re-searching it
    # would be second-guessing NASA's own fit with a coarser tool.
    if inference.is_ambiguous(cnn_prob) and not used_catalog:
        progress("refine", f"Verdict ambiguous (p={cnn_prob:.2f}) — refining "
                            f"period grid around {cand['period']:.4f}d...")
        better = bls_mod.refine_period(time_arr, raw_flux, cand["period"])
        refined_flux = fold_bin_normalize(lc, better["period"], better["epoch"])
        if refined_flux is not None:
            refined_prob = inference.predict_cnn(refined_flux)
            # Keep the refinement only if it actually sharpened the verdict,
            # i.e. moved further from the ambiguous middle.
            if abs(refined_prob - 0.5) > abs(cnn_prob - 0.5):
                cand, flux, cnn_prob = {**cand, **better}, refined_flux, refined_prob
                refined = True
                progress("refine", f"Refinement improved verdict to p={cnn_prob:.2f}")
            else:
                progress("refine", "Refinement did not sharpen the verdict — "
                                    "keeping original period")

    # --- RF (catalog-parameter targets only) --------------------------------
    rf_prob = None
    if used_catalog:
        cat = cand.get("catalog", {})
        row = dict(flux_shape_features(flux))
        row.update({
            "tce_period": cat.get("tce_period"),
            "tce_duration": cat.get("tce_duration"),
            "tce_depth": cat.get("tce_depth"),
            "tce_model_snr": cat.get("tce_model_snr"),
        })
        rf_prob = inference.predict_rf(row)

    # --- Explicit eclipsing-binary check ------------------------------------
    odd_even = bls_mod.odd_even_depth_check(
        time_arr, raw_flux, cand["period"], cand["epoch"],
        cand.get("duration") or 0.1)

    band = inference.confidence_band(cnn_prob)
    alias = cand.get("possible_alias_of")
    decision, reason = _decide(cnn_prob, rf_prob, band, odd_even, alias)

    return {
        "period": cand["period"],
        "epoch": cand["epoch"],
        "duration_days": cand.get("duration"),
        "status": "ok",
        "cnn_probability": round(cnn_prob, 4),
        "rf_probability": round(rf_prob, 4) if rf_prob is not None else None,
        "confidence_band": band,
        "odd_even": odd_even,
        "refined": refined,
        "possible_alias_of": alias,
        "decision": decision,
        "reason": reason,
        "models_agree": (None if rf_prob is None
                          else bool((cnn_prob >= 0.5) == (rf_prob >= 0.5))),
        # The exact 200-bin z-scored folded curve the CNN scored (post-refinement
        # if the period was refined). Served so the dashboard can plot it without
        # re-fetching from MAST. llm._report_payload whitelists its fields, so
        # this never reaches the LLM prompt.
        "folded_curve": [round(float(v), 4) for v in flux],
    }


def _decide(cnn_prob, rf_prob, band, odd_even, alias=None) -> tuple:
    """Accept / reject / escalate.

    Escalation exists because of measured model behaviour, not taste:
      - the CNN is poorly calibrated mid-range (Brier 0.139), so mid-range
        probabilities go to a human rather than being reported as a number
      - the CNN misclassifies likely eclipsing binaries at 23.6% vs RF's 10.9%,
        so a positive-looking verdict with an odd/even flag is escalated
      - when both models are available and disagree, a human decides
      - a positive verdict at a period that is a low-order ratio of a stronger
        candidate is escalated rather than accepted: it may be a BLS alias of
        that signal, or a genuine resonant companion, and the periodogram alone
        cannot tell those apart
    """
    if odd_even.get("is_suspicious") and cnn_prob >= 0.5:
        return ("escalate",
                "Model leans planet candidate, but odd and even transit depths "
                f"differ by {odd_even['sigma']:.1f}sigma — a classic eclipsing "
                "binary signature the CNN is known to miss. Human review required.")

    if rf_prob is not None and ((cnn_prob >= 0.5) != (rf_prob >= 0.5)):
        return ("escalate",
                "CNN and Random Forest disagree on this candidate. Human review "
                "required.")

    if band == "ambiguous":
        return ("escalate",
                "Confidence falls in the band where this model is measurably "
                "overconfident on held-out data. Flagged for human review rather "
                "than reported as a number.")

    if band == "high" and alias:
        return ("escalate",
                f"Transit-like signal, but this period is a {alias['ratio']} ratio "
                f"of the stronger candidate at {alias['period']:.4f}d. It may be a "
                "period alias of that signal rather than a separate body — or a "
                "genuine resonant companion. Human review required.")

    if band == "high":
        return ("accept", "Transit-like signal with high model confidence.")

    return ("reject", "No convincing transit-like signal at this period.")


def fetch_lightcurve(star_id: int):
    """Real MAST fetch. Kept thin so it can be swapped out in tests.

    Falls back to a clean temp cache dir on any Lightkurve error. This matters
    in practice: Lightkurve caches downloads locally and, if a prior download
    was interrupted (killed process, network drop, a previous test run), the
    cached file is corrupt and Lightkurve refuses to read it -- it does not
    automatically detect and re-fetch a bad cache entry. Rather than requiring
    a human to find and delete the right cache folder, we retry once against a
    throwaway directory so a single stale file can never permanently block a
    star. This is deliberately NOT "delete the user's global cache" (that is
    destructive and out of scope for an API call) -- it just stops trusting it
    for this one retry.
    """
    import shutil
    import tempfile

    import lightkurve as lk
    from lightkurve.utils import LightkurveError

    def _do_fetch(download_dir=None):
        search = lk.search_lightcurve(f"KIC {star_id}", mission="Kepler",
                                       cadence="long")
        if search is None or len(search) == 0:
            return None
        collection = (search.download_all(download_dir=download_dir)
                      if download_dir else search.download_all())
        if collection is None or len(collection) == 0:
            return None
        stitched = collection.stitch()
        try:
            stitched.n_quarters = len(collection)
        except Exception:  # noqa: BLE001 - attribute is cosmetic only
            pass
        return stitched

    try:
        return _do_fetch()
    except LightkurveError:
        tmp_dir = tempfile.mkdtemp(prefix="sada_lk_cache_")
        try:
            return _do_fetch(download_dir=tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)