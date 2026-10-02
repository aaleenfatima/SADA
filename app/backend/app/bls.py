"""
BLS period search + odd/even depth check.

This is what makes live inference possible on a star we have no catalog
parameters for: training folded on `tce_period` / `tce_time0bk` from the DR25
catalog, which don't exist for an arbitrary target. We recover them by running
a Box Least Squares search on the raw lightcurve.
"""
from __future__ import annotations

import numpy as np

# Kepler long-cadence practical search range. Below ~0.5d the 29.4-min cadence
# badly undersamples a transit; above ~half the ~4yr baseline you cannot see
# repeats, but we cap lower since demo lightcurves are often partial.
MIN_PERIOD_DAYS = 0.5
MAX_PERIOD_DAYS = 100.0


def search_periods(time, flux, n_candidates: int = 3, duration_grid=None,
                   min_power_frac: float = 0.3):
    """Run BLS and return the top `n_candidates` periods, alias-annotated.

    Returns a list of dicts with period, epoch (t0), duration (days), depth,
    the BLS power statistic, and `possible_alias_of` — the period of a stronger
    candidate to which this one has a small-integer ratio, or None.

    On annotating rather than discarding aliases
    --------------------------------------------
    BLS reports aliases of a true period at integer and rational multiples
    (P/3, 2P, 3P/2, 11P...). The obvious fix is to discard any candidate whose
    period is a small-integer ratio of a stronger one. We deliberately do NOT
    do that, because real multi-planet systems are frequently in orbital
    resonance -- 2:1, 3:2 and 4:3 pairs are common, and Kepler-90 (the system
    AstroNet is known for) contains resonant planets. A period ratio alone
    cannot distinguish a genuine 3:2 resonant companion from a 3:2 BLS alias;
    both look identical in the periodogram.

    So aliases are flagged, not deleted, and the classifier still evaluates
    them. A flagged candidate that folds to a convincing transit at a distinct
    epoch is worth a human's attention; one that does not is easily dismissed.
    Silently deleting them would mean this tool could never find a resonant
    system, which is the opposite of what it is for.

    Candidates below `min_power_frac` of the strongest candidate's power are
    dropped as noise regardless of alias status.
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]
    if len(time) < 100:
        return []

    baseline = time.max() - time.min()
    max_period = min(MAX_PERIOD_DAYS, baseline / 2.0)
    if max_period <= MIN_PERIOD_DAYS:
        return []

    if duration_grid is None:
        duration_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3])

    bls = BoxLeastSquares(time, flux)
    periods = np.linspace(MIN_PERIOD_DAYS, max_period, 20000)
    result = bls.power(periods, duration_grid)

    order = np.argsort(result.power)[::-1]
    candidates = []
    top_power = float(result.power[order[0]])

    for i in order:
        power = float(result.power[i])
        if top_power > 0 and power < min_power_frac * top_power:
            break
        period = float(result.period[i])

        # Skip near-duplicates of an existing candidate (same peak, adjacent
        # grid points) -- these are not aliases, just periodogram width.
        if any(abs(period - c["period"]) / c["period"] < 0.01 for c in candidates):
            continue

        candidates.append({
            "period": period,
            "epoch": float(result.transit_time[i]),
            "duration": float(result.duration[i]),
            "depth": float(result.depth[i]),
            "power": power,
            "possible_alias_of": _alias_of(period, candidates),
        })
        if len(candidates) >= n_candidates:
            break

    return candidates


def _alias_of(period: float, candidates, tol: float = 0.015, max_order: int = 4):
    """Return the period of a stronger candidate this one is a low-order ratio of.

    Annotation only -- see the note in `search_periods` on why aliases are not
    discarded.

    `max_order` is deliberately small. Allowing high-order ratios (13:4, 16:5...)
    makes the rationals dense enough that almost any period matches something
    within tolerance, so every candidate gets flagged and the flag stops meaning
    anything. Low-order ratios are also the ones that actually matter: BLS
    aliases concentrate at 1:2, 1:3, 2:1, 3:1, 3:2, and real orbital resonances
    are low-order too. High-multiple aliases are instead handled by the power
    threshold in `search_periods`, since they are weak.
    """
    for c in candidates:
        ratio = period / c["period"]
        for q in range(1, max_order + 1):
            for p in range(1, max_order + 1):
                target = p / q
                if abs(ratio - target) / target < tol:
                    return {"period": c["period"], "ratio": f"{p}:{q}"}
    return None


def refine_period(time, flux, period: float, window_frac: float = 0.02,
                  n_grid: int = 5000):
    """Re-search a narrow grid around `period` for a better-localized solution.

    Used when a candidate's classification confidence lands in the ambiguous
    band — a slightly-off period smears the transit across bins and depresses
    confidence, so it is worth one refinement pass before giving up.
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    lo = period * (1 - window_frac)
    hi = period * (1 + window_frac)
    grid = np.linspace(lo, hi, n_grid)

    bls = BoxLeastSquares(time, flux)
    result = bls.power(grid, np.array([0.05, 0.1, 0.15, 0.2, 0.3]))
    best = int(np.argmax(result.power))

    return {
        "period": float(result.period[best]),
        "epoch": float(result.transit_time[best]),
        "duration": float(result.duration[best]),
        "depth": float(result.depth[best]),
        "power": float(result.power[best]),
    }


def odd_even_depth_check(time, flux, period: float, epoch: float,
                          duration: float) -> dict:
    """Compare mean depth of odd- vs even-numbered transits.

    A real planet transits to the same depth every orbit. An eclipsing binary
    often alternates (primary/secondary eclipse), so a large odd-even
    difference is the classic EB signature.

    This exists because our own error analysis found the CNN misclassifies
    likely eclipsing binaries at more than double the Random Forest's rate
    (23.6% vs 10.9%) — so we check for it explicitly rather than relying on
    the CNN to have learned it.

    Returns depths, the difference in units of the combined scatter
    (`sigma`), and an `is_suspicious` flag. `sigma` is None when there are too
    few transits of either parity to compare.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[finite], flux[finite]

    epoch_num = np.round((time - epoch) / period)
    phase = (time - epoch) - epoch_num * period
    in_transit = np.abs(phase) < (duration / 2.0)

    baseline = np.median(flux[~in_transit]) if np.any(~in_transit) else np.nan

    odd_mask = in_transit & (epoch_num.astype(int) % 2 != 0)
    even_mask = in_transit & (epoch_num.astype(int) % 2 == 0)

    MIN_POINTS = 5
    if odd_mask.sum() < MIN_POINTS or even_mask.sum() < MIN_POINTS:
        return {"odd_depth": None, "even_depth": None, "sigma": None,
                "is_suspicious": False,
                "note": "too few odd/even transit points to compare"}

    odd_depth = float(baseline - np.mean(flux[odd_mask]))
    even_depth = float(baseline - np.mean(flux[even_mask]))

    odd_err = float(np.std(flux[odd_mask]) / np.sqrt(odd_mask.sum()))
    even_err = float(np.std(flux[even_mask]) / np.sqrt(even_mask.sum()))
    combined_err = np.sqrt(odd_err ** 2 + even_err ** 2)

    sigma = abs(odd_depth - even_depth) / combined_err if combined_err > 0 else None

    return {
        "odd_depth": odd_depth,
        "even_depth": even_depth,
        "sigma": float(sigma) if sigma is not None else None,
        "is_suspicious": bool(sigma is not None and sigma > 3.0),
        "note": None,
    }
