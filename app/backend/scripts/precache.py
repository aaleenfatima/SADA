"""
Precompute the featured demo stars so the demo is instant and survives MAST
being slow or down entirely.

Usage:  python -m scripts.precache
Run once locally (with models + catalog present), commit cache/sada.db.

Star list is intentionally a mix so the UI visibly exercises every code path:
a multi-planet system, known eclipsing binaries (so the odd/even escalation
actually fires on the demo), clean single candidates, and false positives.
Edit FEATURED to taste — verify each ID is what you think it is before
committing, since these are what a reviewer will click first.
"""
import sys
from app import agent, inference, llm, store

FEATURED = [
    11442793,  # Kepler-90 — multi-planet system (the AstroNet discovery)
    10666592,  # HAT-P-7 / Kepler-2 — deep, unambiguous hot Jupiter transit
    6922244,   # Kepler-8 — clean single transiting planet
]

def main():
    store.init_db()
    status = inference.load_models()
    if not status["cnn_loaded"]:
        sys.exit(f"No CNN artifact loaded: {status['errors']}\n"
                 "Export it from Colab first (scripts/export_models_colab.py).")

    import json, os
    catalog = {}
    path = os.environ.get("SADA_CATALOG_PATH", "data/catalog_lookup.json")
    if os.path.exists(path):
        catalog = json.load(open(path))

    for star_id in FEATURED:
        print(f"\n=== KIC {star_id} ===")
        result = agent.analyze_star(
            star_id, catalog_params=catalog.get(str(star_id)),
            progress=lambda s, m: print(f"  [{s}] {m}"))
        if result.get("status") == "complete":
            for cand in result["candidates"]:
                if cand.get("status") == "ok":
                    cand["report"] = llm.generate_report(
                        cand, star_id,
                        rf_unavailable_reason=result.get("rf_unavailable_reason"))
            store.put_cached(star_id, result, is_featured=True)
            print(f"  cached ({len(result['candidates'])} candidates)")
        else:
            print(f"  FAILED: {result.get('error')}")

if __name__ == "__main__":
    main()