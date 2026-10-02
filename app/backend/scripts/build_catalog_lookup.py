"""
Build data/catalog_lookup.json from research/data/balanced_tce.csv.

Unlike the CNN/RF artifacts, this needs no training and no GPU — it is a
straight reshape of a CSV you already have. Run it locally:

    python scripts/build_catalog_lookup.py \
        --csv ../../research/data/balanced_tce.csv \
        --out data/catalog_lookup.json

Produces {kepid: [ {tce_plnt_num, tce_period, tce_time0bk, tce_duration,
tce_depth, tce_model_snr}, ... ]} for every star in the balanced set, so
/analyze can serve real ephemerides (and RF_main) for any of them.

Note: this only covers the 8,074-TCE *balanced* set, not the full 34,032-TCE
DR25 catalog. A star outside this set falls back to live BLS search
automatically — that's expected, not a bug.
"""
import argparse
import json

import pandas as pd

COLUMNS = ["tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration",
           "tce_depth", "tce_model_snr"]


def build(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing expected columns: {missing}")

    before = len(df)
    df = df.dropna(subset=COLUMNS)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing catalog values")

    lookup = {}
    for kepid, group in df.groupby("kepid"):
        lookup[str(int(kepid))] = group[COLUMNS].to_dict("records")

    with open(out_path, "w") as f:
        json.dump(lookup, f)

    print(f"Wrote {out_path}: {len(lookup):,} stars, {len(df):,} TCEs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../../research/data/balanced_tce.csv")
    p.add_argument("--out", default="data/catalog_lookup.json")
    args = p.parse_args()
    build(args.csv, args.out)
