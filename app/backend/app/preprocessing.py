"""
Preprocessing — MUST stay byte-identical to the training pipeline.

Source of truth: research/notebooks/02_eda.ipynb, `preprocess_lightcurve()`.
Training did, in exactly this order:
    1. search_lightcurve(KIC, mission="Kepler", cadence="long").download_all()
    2. .stitch().remove_outliers(sigma=3)
    3. .fold(period=period, epoch_time=epoch)
    4. .bin(bins=200)
    5. flux -> np.array(filled with nan); reject if ANY nan
    6. z-score: (flux - mean) / std
    7. reject if len != 200

Any change here silently invalidates the served model. Do not "improve" this
function without retraining.
"""
from __future__ import annotations

import numpy as np

N_BINS = 200
SIGMA_CLIP = 3


def fold_bin_normalize(lc_stitched, period: float, epoch: float, n_bins: int = N_BINS):
    """Steps 2-7 above, given an already-stitched lightkurve object.

    Returns a (n_bins,) float32 array, or None if the star fails preprocessing
    (same rejection rules as training: any NaN, or wrong length).
    """
    lc = lc_stitched.remove_outliers(sigma=SIGMA_CLIP)
    folded = lc.fold(period=period, epoch_time=epoch)
    binned = folded.bin(bins=n_bins)

    flux = np.array(binned.flux.filled(fill_value=np.nan))

    if np.any(np.isnan(flux)):
        return None

    std = np.std(flux)
    if std == 0:
        return None
    flux = (flux - np.mean(flux)) / std

    if len(flux) != n_bins:
        return None

    return flux.astype(np.float32)


def flux_shape_features(flux) -> dict:
    """The 9 engineered flux features used by RF_main.

    Source of truth: research/notebooks/03_baseline_model.ipynb.
    In-transit window = central third of the folded array (crude but must match
    training exactly).
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    mean, std = flux.mean(), flux.std()
    minimum, maximum = flux.min(), flux.max()

    lo, hi = n // 3, 2 * n // 3
    in_transit = flux[lo:hi]
    out_transit = np.concatenate([flux[:lo], flux[hi:]])

    depth_est = out_transit.mean() - in_transit.min()
    in_std = in_transit.std()
    out_std = out_transit.std() if len(out_transit) else 0.0

    if std > 0:
        skew = ((flux - mean) ** 3).mean() / (std ** 3)
        kurt = ((flux - mean) ** 4).mean() / (std ** 4) - 3
    else:
        skew, kurt = 0.0, 0.0

    return {
        "flux_mean": mean,
        "flux_std": std,
        "flux_min": minimum,
        "flux_max": maximum,
        "flux_depth_est": depth_est,
        "flux_in_transit_std": in_std,
        "flux_out_transit_std": out_std,
        "flux_skew": skew,
        "flux_kurtosis": kurt,
    }


# Feature order for RF_main — MUST match training column order exactly.
BLS_FEATURES = ["tce_period", "tce_duration", "tce_depth", "tce_model_snr"]
FLUX_FEATURES = [
    "flux_mean", "flux_std", "flux_min", "flux_max", "flux_depth_est",
    "flux_in_transit_std", "flux_out_transit_std", "flux_skew", "flux_kurtosis",
]
RF_FEATURES = BLS_FEATURES + FLUX_FEATURES
