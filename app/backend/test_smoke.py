"""Smoke test: exercises the full API path with a synthetic lightcurve and a
stubbed CNN, so no MAST, no torch weights, and no network are required."""
import os, sys, time, types
import numpy as np

os.environ["SADA_DB_PATH"] = "/tmp/test_sada.db"
if os.path.exists("/tmp/test_sada.db"): os.remove("/tmp/test_sada.db")

from app import inference, agent, store, llm

# --- Build a synthetic lightcurve with a REAL injected transit -------------
rng = np.random.default_rng(42)
PERIOD, EPOCH, DUR, DEPTH = 3.5, 2.0, 0.12, 0.01
t = np.arange(0, 400, 0.0204)           # ~29.4 min Kepler long cadence
f = 1.0 + rng.normal(0, 0.0005, len(t))
phase = (t - EPOCH + PERIOD/2) % PERIOD - PERIOD/2
f[np.abs(phase) < DUR/2] -= DEPTH

class FakeArr:
    def __init__(self, v): self.value = v
class FakeLC:
    """Minimal stand-in exposing what the agent + preprocessing actually touch."""
    def __init__(self, t, f):
        self.time, self.flux, self.n_quarters = FakeArr(t), FakeArr(f), 14
    def remove_outliers(self, sigma=3): return self
    def fold(self, period, epoch_time):
        ph = (self.time.value - epoch_time + period/2) % period - period/2
        o = np.argsort(ph)
        return FakeLC(ph[o], self.flux.value[o])
    def bin(self, bins):
        idx = np.linspace(0, len(self.flux.value), bins+1).astype(int)
        b = np.array([self.flux.value[idx[i]:idx[i+1]].mean() for i in range(bins)])
        out = FakeLC(np.zeros(bins), b)
        out.flux = types.SimpleNamespace(filled=lambda fill_value: b)
        return out

lc = FakeLC(t, f)

# --- Stub the CNN: high prob when the folded array has a real dip ----------
def fake_predict_cnn(flux):
    flux = np.asarray(flux)
    depth = flux.mean() - flux.min()
    return float(min(0.98, max(0.02, (depth - 1.0) / 2.5)))
inference.predict_cnn = fake_predict_cnn
inference.models_status = lambda: {"cnn_loaded": True, "rf_loaded": False, "errors": {}}

print("=== 1. preprocessing parity ===")
from app.preprocessing import fold_bin_normalize, flux_shape_features
arr = fold_bin_normalize(lc, PERIOD, EPOCH)
assert arr is not None and len(arr) == 200, "preprocessing returned bad array"
assert abs(arr.mean()) < 1e-4 and abs(arr.std() - 1) < 1e-4, "not z-scored"
print(f"  OK: shape={arr.shape} mean={arr.mean():.2e} std={arr.std():.4f}")
print(f"  transit visible: min={arr.min():.2f} at bin {int(np.argmin(arr))}")

print("=== 2. flux shape features ===")
feats = flux_shape_features(arr)
assert len(feats) == 9
print(f"  OK: 9 features, depth_est={feats['flux_depth_est']:.3f}")

print("=== 3. BLS period recovery ===")
from app import bls
cands = bls.search_periods(t, f, n_candidates=3)
assert cands, "BLS found nothing"
best = cands[0]["period"]
err = abs(best - PERIOD) / PERIOD
print(f"  injected={PERIOD}d  recovered={best:.4f}d  error={err*100:.2f}%")
assert err < 0.02, f"BLS failed to recover injected period (got {best})"
print(f"  OK: {len(cands)} candidates, harmonics de-duplicated")

print("=== 4. odd/even check ===")
oe = bls.odd_even_depth_check(t, f, PERIOD, EPOCH, DUR)
print(f"  sigma={oe['sigma']}, suspicious={oe['is_suspicious']} (equal depths -> expect False)")
assert oe["is_suspicious"] is False, "flagged a clean transit as an EB"

print("=== 5. full agent loop (live-search path) ===")
msgs = []
res = agent.analyze_star(11111111, catalog_params=None,
                         progress=lambda s, m: msgs.append((s, m)),
                         fetch_fn=lambda sid: lc)
assert res["status"] == "complete", res
assert res["ephemeris_source"] == "live_bls_search"
assert res["rf_available"] is False and res["rf_unavailable_reason"]
print(f"  candidates: {len(res['candidates'])}, elapsed={res['elapsed_seconds']}s")
for c in res["candidates"]:
    print(f"    P={c['period']:.4f}d band={c['confidence_band']} decision={c['decision']}")
print("  progress stages:", [s for s, _ in msgs])

print("=== 6. catalog path (RF attempted) ===")
res2 = agent.analyze_star(11111111,
    catalog_params=[{"tce_period": PERIOD, "tce_time0bk": EPOCH,
                     "tce_duration": DUR*24, "tce_depth": 10000.0,
                     "tce_model_snr": 30.0}],
    progress=lambda s, m: None, fetch_fn=lambda sid: lc)
assert res2["ephemeris_source"] == "dr25_catalog"
# rf_available requires BOTH catalog params present AND the RF artifact loaded.
# This test's stub reports rf_loaded=False, so even with a real catalog entry,
# rf_available must correctly be False -- asserting True here would be testing
# the bug this was fixed from (rf_available used to mean only "catalog
# existed", which produced a false "catalog not available" report when the
# real reason was a missing model artifact).
assert res2["rf_available"] is False
assert res2["rf_unavailable_reason"] and "model artifact" in res2["rf_unavailable_reason"]
print(f"  OK: rf_probability={res2['candidates'][0]['rf_probability']} (None = RF artifact absent, expected here)")

print("=== 7. report generation (template fallback, no API key) ===")
rep = llm.generate_report(res["candidates"][0], 11111111)
assert rep["source"] == "template" and len(rep["text"]) > 40
print(f"  OK [{rep['source']}]: {rep['text'][:150]}...")

print("=== 8. API endpoints ===")
from fastapi.testclient import TestClient
from app import main
main.inference = inference
import app.agent as ag
ag.fetch_lightcurve = lambda sid: lc
with TestClient(main.app) as client:
    h = client.get("/healthz").json()
    print(f"  /healthz ready={h['ready']}")
    r = client.post("/analyze", json={"star_id": 11111111}).json()
    assert r["job_id"], r
    print(f"  /analyze -> job_id={r['job_id']} cached={r['cached']}")
    for _ in range(50):
        job = client.get(f"/jobs/{r['job_id']}").json()
        if job["status"] in ("complete", "failed"): break
        time.sleep(0.2)
    assert job["status"] == "complete", job.get("error")
    print(f"  /jobs -> status={job['status']}, {len(job['progress'])} progress entries")
    r2 = client.post("/analyze", json={"star_id": 11111111}).json()
    assert r2["cached"] is True, "second call should hit cache"
    print(f"  cache hit on repeat call: {r2['cached']}")

print("\nALL CHECKS PASSED")