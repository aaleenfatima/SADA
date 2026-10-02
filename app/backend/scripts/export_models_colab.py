"""
Run this in Colab (after training) to export the artifacts the API needs.

The repo currently has NO saved model weights — research/models/ contains only
result CSVs. The API will refuse to serve predictions until these exist, rather
than fabricating output.

Produces:
    models/cnn_1d.pt            CNN state_dict (3,953 params)
    models/rf_main.joblib       fitted RandomForestClassifier
    data/catalog_lookup.json    kepid -> DR25 TCE params, for the catalog path

Copy all three into the API repo before deploying.
"""

EXPORT_SNIPPET = r'''
# ---- paste into Colab after your models are trained ----
import os, json, torch, joblib

OUT = '/content/drive/MyDrive/SADA/export'
os.makedirs(OUT, exist_ok=True)

# 1. CNN weights. `model` must be the trained 8->16->32 Conv1DClassifier.
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert n_params == 3953, f"Expected 3,953 params, got {n_params:,} — wrong architecture!"
torch.save(model.state_dict(), f'{OUT}/cnn_1d.pt')

# 2. RF_main. `rf_main` must be fitted on RF_FEATURES in this exact order.
RF_FEATURES = ['tce_period','tce_duration','tce_depth','tce_model_snr',
               'flux_mean','flux_std','flux_min','flux_max','flux_depth_est',
               'flux_in_transit_std','flux_out_transit_std','flux_skew','flux_kurtosis']
assert list(getattr(rf_main, 'feature_names_in_', RF_FEATURES)) == RF_FEATURES, \
    "RF feature order does not match the API's expected order!"
joblib.dump(rf_main, f'{OUT}/rf_main.joblib')

# 3. Catalog lookup: every TCE per star, so the API can serve known targets
#    with real ephemerides (and therefore RF).
cols = ['tce_plnt_num','tce_period','tce_time0bk','tce_duration','tce_depth','tce_model_snr']
lookup = {}
for kepid, grp in balanced.groupby('kepid'):
    lookup[str(int(kepid))] = grp[cols].to_dict('records')
with open(f'{OUT}/catalog_lookup.json', 'w') as f:
    json.dump(lookup, f)

print(f"Exported to {OUT}:")
print(f"  cnn_1d.pt ({n_params:,} params)")
print(f"  rf_main.joblib")
print(f"  catalog_lookup.json ({len(lookup):,} stars)")
'''

if __name__ == "__main__":
    print(__doc__)
    print(EXPORT_SNIPPET)
