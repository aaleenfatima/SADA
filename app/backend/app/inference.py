"""
Model loading + inference.

Two models are served:
  - CNN_1D  : needs only the 200-point folded flux array. Works on ANY star.
  - RF_main : needs 4 catalog features (period/duration/depth/SNR) in the units
              and scale they had in the DR25 catalog, plus 9 flux features.

IMPORTANT — why RF is not served in live-search mode:
RF_main was trained on catalog `tce_period` (days), `tce_duration` (hours),
`tce_depth` (ppm) and `tce_model_snr` (a Kepler-pipeline-specific statistic).
A live BLS search produces its own period/duration/depth in different units and
on a different scale, and no equivalent of model SNR at all. Feeding BLS-derived
analogues into RF would be a silent distribution shift, and its output would not
mean what the training metrics say it means. So RF is served ONLY for stars
where real catalog parameters exist. This is surfaced in the API response rather
than papered over.
"""
from __future__ import annotations

import json
import os

import numpy as np

MODELS_DIR = os.environ.get("SADA_MODELS_DIR", "models")

_cnn = None
_rf = None
_load_errors: dict = {}


class Conv1DClassifier:
    """Architecture must match training exactly (3,953 params).

    Defined lazily inside a function so torch is only imported when actually
    serving the CNN.
    """


def _unwrap_state_dict(raw):
    """Accept a raw state_dict OR a wrapped checkpoint dict.

    We don't know how `cnn_1d_best.pt` was actually saved without opening it on
    the machine that created it, so this checks the common wrapper key names
    (`model_state_dict`, `state_dict`, `model`) before assuming `raw` IS the
    state dict. If none of those keys exist and `raw` isn't a plain tensor dict
    either, this raises rather than silently loading garbage.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Checkpoint is a {type(raw).__name__}, not a dict")

    for key in ("model_state_dict", "state_dict", "model"):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]

    # Heuristic: a real state_dict's values are all tensors.
    import torch
    if raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw

    raise ValueError(
        f"Checkpoint dict has keys {list(raw.keys())[:10]} — none look like a "
        "state_dict or a known wrapper key (model_state_dict/state_dict/model). "
        "Check how this checkpoint was saved.")


def _build_cnn():
    import torch.nn as nn

    class _Conv1DClassifier(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv1d(1, 8, 5, padding=2), nn.BatchNorm1d(8), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(dropout))
            self.block2 = nn.Sequential(
                nn.Conv1d(8, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(dropout))
            self.block3 = nn.Sequential(
                nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(dropout))
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.gap(x).squeeze(-1)
            return self.head(x).squeeze(-1)

    return _Conv1DClassifier()


def load_models():
    """Load whatever artifacts are present. Missing artifacts are recorded, not fatal."""
    global _cnn, _rf, _load_errors
    _load_errors = {}

    # Accept either filename — training checkpoints are often saved as
    # "*_best.pt" (the state captured at the best validation-loss epoch).
    cnn_candidates = [
        os.path.join(MODELS_DIR, "cnn_1d.pt"),
        os.path.join(MODELS_DIR, "cnn_1d_best.pt"),
    ]
    cnn_path = next((p for p in cnn_candidates if os.path.exists(p)), None)
    try:
        import torch
        if cnn_path is None:
            raise FileNotFoundError(
                f"None of {cnn_candidates} found — export a checkpoint "
                "(see scripts/export_models_colab.py) or point SADA_MODELS_DIR "
                "at research/models.")
        raw = torch.load(cnn_path, map_location="cpu")
        state = _unwrap_state_dict(raw)
        model = _build_cnn()
        model.load_state_dict(state)
        model.eval()
        _cnn = model
    except Exception as e:  # noqa: BLE001 - surfaced via /healthz
        _cnn = None
        _load_errors["cnn"] = str(e)

    rf_path = os.path.join(MODELS_DIR, "rf_main.joblib")
    try:
        import joblib
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"{rf_path} not found — export it from Colab "
                                     "(see scripts/export_models_colab.py)")
        _rf = joblib.load(rf_path)
    except Exception as e:  # noqa: BLE001
        _rf = None
        _load_errors["rf"] = str(e)

    return {"cnn_loaded": _cnn is not None, "rf_loaded": _rf is not None,
            "errors": _load_errors}


def models_status() -> dict:
    return {"cnn_loaded": _cnn is not None, "rf_loaded": _rf is not None,
            "errors": _load_errors}


def predict_cnn(flux) -> float:
    """Probability that this folded lightcurve is a planet candidate."""
    if _cnn is None:
        raise RuntimeError(
            "CNN not loaded: " + _load_errors.get("cnn", "unknown error"))
    import torch

    x = torch.tensor(np.asarray(flux, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logit = _cnn(x)
        prob = torch.sigmoid(logit).item()
    return float(prob)


def predict_rf(feature_row: dict):
    """RF_main probability. `feature_row` must contain all 13 RF_FEATURES.

    Returns None if RF is unavailable or any catalog feature is missing — the
    caller surfaces that as `rf_available: false` rather than guessing values.
    """
    from .preprocessing import RF_FEATURES

    if _rf is None:
        return None
    missing = [f for f in RF_FEATURES if feature_row.get(f) is None]
    if missing:
        return None

    X = np.array([[feature_row[f] for f in RF_FEATURES]], dtype=float)
    return float(_rf.predict_proba(X)[0, 1])


# --- Confidence banding -----------------------------------------------------
# Grounded in measured calibration on the held-out test set (n=678):
#   RF_main  Brier = 0.083  (reasonably calibrated)
#   CNN_1D   Brier = 0.139  (overconfident between roughly p=0.4 and p=0.8)
# Because of that, the CNN's raw probability is NOT shown to users as a
# percentage. It is bucketed, and the ambiguous band is deliberately wide over
# exactly the range where the reliability diagram showed the CNN drifting below
# the diagonal.

AMBIGUOUS_LOW = 0.40
AMBIGUOUS_HIGH = 0.80


def confidence_band(prob: float) -> str:
    if prob >= AMBIGUOUS_HIGH:
        return "high"
    if prob < AMBIGUOUS_LOW:
        return "low"
    return "ambiguous"


def is_ambiguous(prob: float) -> bool:
    return confidence_band(prob) == "ambiguous"
