"""Canonical grouped-split rerun for the SADA paper.

Single source of truth for every number in the paper:
  * 70/15/15 split that keeps all TCEs of a star (kepid) in one partition
  * 5 model seeds for each CNN config, per-example test predictions saved
  * all outputs written to research/results/

Stages (each resumable and independent, so they can run in parallel processes):
  tabular   LR_catalog, RF_main, RF_with_robovetter
  cnn       CNN_1D, CNN_1D_cosine
  dual      CNN_dualview members + 5-model ensembles
  aug       noise-augmentation sweep (alpha > 0; alpha = 0 is CNN_1D)
  lc        learning curves, fractions < 1 (fraction 1.0 is CNN_1D / RF_main)
  cv        5-fold StratifiedGroupKFold cross-validation
  aggregate merge everything into metrics_summary.csv

Usage:  python run_grouped.py <stage> [--smoke]
"""
import argparse, os, re, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'data', 'balanced_tce.csv')
NPY_DIR = os.path.join(HERE, 'data', 'processed')

SPLIT_SEED = 42
MODEL_SEEDS = [42, 43, 44, 45, 46]          # one run per seed for every CNN config
DUAL_SEEDS = list(range(42, 67))            # 25 members -> 5 disjoint ensembles of 5
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5]         # alpha = 0 is CNN_1D
FRACTIONS = [0.10, 0.25, 0.50, 0.75]        # fraction 1.0 is CNN_1D / RF_main
N_EPOCHS, PATIENCE, BATCH = 60, 8, 64
NAT_PC_FRAC = 1 / 8                         # natural PC:FP ratio of ~1:7 (from project notes)

BLS_FEATURES = ['tce_period', 'tce_duration', 'tce_depth', 'tce_model_snr']
FLUX_FEATURES = ['flux_mean', 'flux_std', 'flux_min', 'flux_max', 'flux_depth_est',
                 'flux_in_transit_std', 'flux_out_transit_std', 'flux_skew', 'flux_kurtosis']
RF_FEATURES = BLS_FEATURES + FLUX_FEATURES
RF_RV_FEATURES = RF_FEATURES + ['robovetter_score']

torch.set_num_threads(2)


# --------------------------------------------------------------------------- data
def flux_shape_features(flux):
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    mean, std = flux.mean(), flux.std()
    lo, hi = n // 3, 2 * n // 3
    inn, out = flux[lo:hi], np.concatenate([flux[:lo], flux[hi:]])
    if std > 0:
        skew = ((flux - mean) ** 3).mean() / std ** 3
        kurt = ((flux - mean) ** 4).mean() / std ** 4 - 3
    else:
        skew = kurt = 0.0
    return {'flux_mean': mean, 'flux_std': std, 'flux_min': flux.min(), 'flux_max': flux.max(),
            'flux_depth_est': out.mean() - inn.min(), 'flux_in_transit_std': inn.std(),
            'flux_out_transit_std': out.std() if len(out) else 0.0,
            'flux_skew': skew, 'flux_kurtosis': kurt}


def load_data():
    pat = re.compile(r'^(\d+)_(\d+)_(PC|FP)\.npy$')
    recs = []
    for f in sorted(os.listdir(NPY_DIR)):
        m = pat.match(f)
        if m:
            k, p, l = m.groups()
            recs.append({'kepid': int(k), 'tce_plnt_num': int(p), 'label': int(l == 'PC'),
                         'npy_path': os.path.join(NPY_DIR, f)})
    idx = pd.DataFrame(recs)
    data = idx.merge(pd.read_csv(CSV_PATH), on=['kepid', 'tce_plnt_num'], how='inner')
    data = data.dropna(subset=BLS_FEATURES).reset_index(drop=True)
    assert ((data['label'] == 1) == (data['disposition'] == 'PC')).all()
    flux = np.stack([np.load(p).astype(np.float32) for p in data['npy_path']])
    data = pd.concat([data, pd.DataFrame([flux_shape_features(f) for f in flux])], axis=1)
    data.index.name = 'row_id'
    return data, flux


def make_split(data):
    """70/15/15, stratified by label, grouped by kepid (20 grouped folds: 3 test, 3 val, 14 train)."""
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SPLIT_SEED)
    fold = np.empty(len(data), dtype=int)
    for i, (_, te) in enumerate(sgkf.split(data, data['label'], groups=data['kepid'])):
        fold[te] = i
    split = np.where(fold < 3, 'test', np.where(fold < 6, 'val', 'train'))
    tr, va, te = (np.where(split == s)[0] for s in ('train', 'val', 'test'))
    k = data['kepid'].values
    assert not (set(k[tr]) & set(k[te])) and not (set(k[tr]) & set(k[va])) and not (set(k[va]) & set(k[te]))
    return tr, va, te, split


# ------------------------------------------------------------------------ metrics
def metrics(y, p):
    pred = (p > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sd = lambda a, b: a / b if b else float('nan')
    prec, rec = sd(tp, tp + fp), sd(tp, tp + fn)
    fpr = sd(fp, fp + tn)
    pi = NAT_PC_FRAC   # precision/F1/accuracy re-weighted to natural prevalence (recall, FPR, AUC unchanged)
    prec_nat = sd(pi * rec, pi * rec + (1 - pi) * fpr)
    return {'accuracy': (tp + tn) / len(y), 'precision': prec, 'recall': rec,
            'f1': sd(2 * prec * rec, prec + rec), 'roc_auc': roc_auc_score(y, p),
            'brier': brier_score_loss(y, p), 'fpr': fpr,
            'precision_nat': prec_nat, 'f1_nat': sd(2 * prec_nat * rec, prec_nat + rec),
            'accuracy_nat': pi * rec + (1 - pi) * (1 - fpr), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


class Recorder:
    """Appends one row per (config, seed) to runs_<stage>.csv and test predictions to preds_<stage>.csv."""
    def __init__(self, stage, outdir, te_idx, y_test):
        self.rp = os.path.join(outdir, f'runs_{stage}.csv')
        self.pp = os.path.join(outdir, f'preds_{stage}.csv')
        self.te, self.y = te_idx, y_test
        self.done = set()
        if os.path.exists(self.rp):
            r = pd.read_csv(self.rp)
            self.done = set(zip(r['config'], r['seed']))

    def has(self, config, seed):
        return (config, seed) in self.done

    def add(self, config, seed, prob, extra=None):
        row = {'config': config, 'seed': seed, **metrics(self.y, prob), **(extra or {})}
        pd.DataFrame([row]).to_csv(self.rp, mode='a', header=not os.path.exists(self.rp), index=False)
        pd.DataFrame({'config': config, 'seed': seed, 'row_id': self.te, 'y_true': self.y,
                      'prob': prob}).to_csv(self.pp, mode='a', header=not os.path.exists(self.pp), index=False)
        self.done.add((config, seed))
        print(f"  {config:28s} seed={seed:<3d} f1={row['f1']:.4f} auc={row['roc_auc']:.4f}", flush=True)


# -------------------------------------------------------------------------- models
class CNN1D(nn.Module):
    def __init__(self, d=0.3):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(),
                                         nn.MaxPool1d(2), nn.Dropout(d))
        self.f = nn.Sequential(blk(1, 8), blk(8, 16), blk(16, 32))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(d), nn.Linear(16, 1))

    def forward(self, x):
        return self.head(self.gap(self.f(x)).squeeze(-1)).squeeze(-1)


class DualView(nn.Module):
    LOCAL = slice(80, 120)   # 40-point crop around bin 100, as in notebook 06

    def __init__(self, d=0.3):
        super().__init__()
        def br():
            blk = lambda i, o: [nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(),
                                nn.MaxPool1d(2), nn.Dropout(d)]
            return nn.Sequential(*blk(1, 8), *blk(8, 16))
        self.g, self.l = br(), br()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(d), nn.Linear(16, 1))

    def forward(self, x):
        g = self.gap(self.g(x)).squeeze(-1)
        l = self.gap(self.l(x[:, :, self.LOCAL])).squeeze(-1)
        return self.head(torch.cat([g, l], 1)).squeeze(-1)


class CNNHybrid(nn.Module):
    """CNN_1D trunk on the 200-bin flux; the 4 standardized catalog statistics are concatenated to the pooled features.
    Input x has shape (N, 1, 204): 200 flux bins followed by the 4 catalog values."""
    def __init__(self, d=0.3):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(),
                                         nn.MaxPool1d(2), nn.Dropout(d))
        self.f = nn.Sequential(blk(1, 8), blk(8, 16), blk(16, 32))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(32 + len(BLS_FEATURES), 16), nn.ReLU(), nn.Dropout(d), nn.Linear(16, 1))

    def forward(self, x):
        z = self.gap(self.f(x[:, :, :200])).squeeze(-1)
        return self.head(torch.cat([z, x[:, 0, 200:]], 1)).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_predict(make_model, X, y, tr, va, te, seed, alpha=0.0, cosine=False, epochs=None, hist=None):
    """Same protocol as notebooks 04a/06: AdamW 1e-3, wd 1e-4, batch 64, early stop on val loss."""
    epochs = epochs or N_EPOCHS
    torch.manual_seed(seed)
    model = make_model()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS) if cosine else None
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    best, state, bad = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        tl = []
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb = Xtr[b]
            if alpha > 0:
                xb = xb + alpha * xb.std(dim=-1, keepdim=True, unbiased=False) * torch.randn_like(xb)
            opt.zero_grad()
            loss = crit(model(xb), ytr[b])
            loss.backward()
            opt.step()
            tl.append(loss.item())
        if sched:
            sched.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva), yva).item()
        if hist is not None:
            hist.append((float(np.mean(tl)), vl))
        if vl < best:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X[te])).numpy()


# -------------------------------------------------------------------------- stages
def fit_tabular(model, cols, data, tr, te, scale=False):
    Xtr, Xte = data.loc[tr, cols].values, data.loc[te, cols].values
    if scale:
        s = StandardScaler().fit(Xtr)
        Xtr, Xte = s.transform(Xtr), s.transform(Xte)
    model.fit(Xtr, data.loc[tr, 'label'].values)
    return model.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['tabular', 'cnn', 'hybrid', 'dual', 'aug', 'lc', 'cv', 'curves', 'aggregate'])
    ap.add_argument('--smoke', action='store_true', help='1 seed, 2 epochs, results_smoke/')
    a = ap.parse_args()

    global MODEL_SEEDS, DUAL_SEEDS, N_EPOCHS
    outdir = os.path.join(HERE, 'results_smoke' if a.smoke else 'results')
    os.makedirs(outdir, exist_ok=True)
    if a.smoke:
        MODEL_SEEDS, DUAL_SEEDS, N_EPOCHS = MODEL_SEEDS[:1], DUAL_SEEDS[:5], 2

    if a.stage == 'aggregate':
        return aggregate(outdir)

    t0 = time.time()
    data, flux = load_data()
    tr, va, te, split = make_split(data)
    y = data['label'].values
    if a.stage == 'tabular':   # write the audit file once
        data.assign(split=split)[['kepid', 'tce_plnt_num', 'label', 'split']].to_csv(
            os.path.join(outdir, 'split.csv'))
    print(f"rows={len(data)} stars={data['kepid'].nunique()} train/val/test={len(tr)}/{len(va)}/{len(te)} "
          f"test PC/FP={int(y[te].sum())}/{int((1 - y[te]).sum())} (load {time.time() - t0:.0f}s)", flush=True)

    X = torch.tensor(flux).unsqueeze(1)
    yt = torch.tensor(y.astype(np.float32))
    rec = Recorder(a.stage, outdir, te, y[te])

    if a.stage == 'tabular':
        if not rec.has('LR_catalog', 0):
            rec.add('LR_catalog', 0, fit_tabular(LogisticRegression(max_iter=1000), BLS_FEATURES, data, tr, te, True))
        for s in MODEL_SEEDS:
            for name, cols in (('RF_main', RF_FEATURES), ('RF_with_robovetter', RF_RV_FEATURES),
                               ('RF_catalog_only', BLS_FEATURES), ('RF_flux_only', FLUX_FEATURES)):
                if not rec.has(name, s):
                    rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=2)
                    rec.add(name, s, fit_tabular(rf, cols, data, tr, te))

    elif a.stage == 'cnn':
        for s in MODEL_SEEDS:
            for name, kw in (('CNN_1D', {}), ('CNN_1D_cosine', {'cosine': True})):
                if not rec.has(name, s):
                    rec.add(name, s, train_predict(CNN1D, X, yt, tr, va, te, s, **kw), {'n_params': n_params(CNN1D())})

    elif a.stage == 'hybrid':   # catalog statistics: signed log1p, then standardized with training-set mean/std
        v = data[BLS_FEATURES].values.astype(np.float64)   # model SNR contains -1 sentinels, so use a signed log
        cat = np.sign(v) * np.log1p(np.abs(v))
        cat = (cat - cat[tr].mean(0)) / cat[tr].std(0)
        Xh = torch.cat([X, torch.tensor(cat, dtype=torch.float32).unsqueeze(1)], dim=2)
        for s in MODEL_SEEDS:
            if not rec.has('CNN_1D_catalog', s):
                rec.add('CNN_1D_catalog', s, train_predict(CNNHybrid, Xh, yt, tr, va, te, s), {'n_params': n_params(CNNHybrid())})

    elif a.stage == 'dual':
        for s in DUAL_SEEDS:
            if not rec.has('CNN_dualview', s):
                rec.add('CNN_dualview', s, train_predict(DualView, X, yt, tr, va, te, s, cosine=True),
                        {'n_params': n_params(DualView())})
        preds = pd.read_csv(rec.pp)
        for r in range(len(DUAL_SEEDS) // 5):
            seeds = DUAL_SEEDS[5 * r:5 * r + 5]
            if not rec.has('CNN_dualview_ens5', r):
                sub = preds[(preds.config == 'CNN_dualview') & preds.seed.isin(seeds)]
                rec.add('CNN_dualview_ens5', r, sub.groupby('row_id')['prob'].mean().reindex(te).values)

    elif a.stage == 'aug':
        for a_ in ALPHAS:
            for s in MODEL_SEEDS:
                name = f'CNN_1D_aug_a{a_:.2f}'
                if not rec.has(name, s):
                    rec.add(name, s, train_predict(CNN1D, X, yt, tr, va, te, s, alpha=a_))

    elif a.stage == 'lc':
        for f in FRACTIONS:
            for s in MODEL_SEEDS:
                sub, _ = train_test_split(tr, train_size=f, random_state=s, stratify=y[tr])
                n1, n2 = f'LC_CNN_1D_f{f:.2f}', f'LC_RF_main_f{f:.2f}'
                if not rec.has(n1, s):
                    rec.add(n1, s, train_predict(CNN1D, X, yt, sub, va, te, s), {'n_train': len(sub)})
                if not rec.has(n2, s):
                    rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=2)
                    rec.add(n2, s, fit_tabular(rf, RF_FEATURES, data, sub, te), {'n_train': len(sub)})

    elif a.stage == 'curves':   # CNN_1D seed 42 again, keeping the loss history; must reproduce the saved predictions
        hist = []
        p = train_predict(CNN1D, X, yt, tr, va, te, MODEL_SEEDS[0], hist=hist)
        pd.DataFrame(hist, columns=['train_loss', 'val_loss']).rename_axis('epoch').to_csv(
            os.path.join(outdir, 'cnn_training_history.csv'))
        ref = pd.read_csv(os.path.join(outdir, 'preds_cnn.csv'))
        ref = ref[(ref.config == 'CNN_1D') & (ref.seed == MODEL_SEEDS[0])].sort_values('row_id')['prob'].values
        print(f"max |prob - saved prob| = {np.abs(p - ref).max():.2e}; epochs run = {len(hist)}")

    elif a.stage == 'cv':
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
        rows = []
        for fold, (tr_f, te_f) in enumerate(sgkf.split(data, y, groups=data['kepid']), 1):
            assert not (set(data['kepid'].values[tr_f]) & set(data['kepid'].values[te_f]))
            inner_tr, inner_va = train_test_split(tr_f, test_size=0.15, random_state=SPLIT_SEED, stratify=y[tr_f])
            # val rows may share stars with inner_tr (early stopping only; test fold is star-disjoint)
            outs = {
                'LR_catalog': fit_tabular(LogisticRegression(max_iter=1000), BLS_FEATURES, data, tr_f, te_f, True),
                'RF_main': fit_tabular(RandomForestClassifier(n_estimators=300, random_state=SPLIT_SEED, n_jobs=2),
                                       RF_FEATURES, data, tr_f, te_f),
                'RF_with_robovetter': fit_tabular(RandomForestClassifier(n_estimators=300, random_state=SPLIT_SEED, n_jobs=2),
                                                  RF_RV_FEATURES, data, tr_f, te_f),
                'CNN_1D': train_predict(CNN1D, X, yt, inner_tr, inner_va, te_f, SPLIT_SEED),
            }
            for name, p in outs.items():
                rows.append({'model': name, 'fold': fold, **metrics(y[te_f], p)})
                print(f"  fold {fold} {name:20s} f1={rows[-1]['f1']:.4f} auc={rows[-1]['roc_auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(outdir, 'cv_fold_results.csv'), index=False)

    print(f"stage {a.stage} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def aggregate(outdir):
    runs = pd.concat([pd.read_csv(os.path.join(outdir, f)) for f in sorted(os.listdir(outdir))
                      if f.startswith('runs_')], ignore_index=True)
    # aliases so each sweep/curve is complete while every config is computed exactly once
    alias = []
    for src, dst in [('CNN_1D', 'CNN_1D_aug_a0.00'), ('CNN_1D', 'LC_CNN_1D_f1.00'), ('RF_main', 'LC_RF_main_f1.00')]:
        alias.append(runs[runs.config == src].assign(config=dst))
    runs = pd.concat([runs] + alias, ignore_index=True)
    runs.to_csv(os.path.join(outdir, 'metrics_per_run.csv'), index=False)
    cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier', 'precision_nat', 'f1_nat', 'accuracy_nat']
    g = runs.groupby('config')
    summ = g[cols].mean().add_suffix('_mean').join(g[cols].std().add_suffix('_std')).join(g.size().rename('n_runs'))
    summ = summ[[c for k in cols for c in (f'{k}_mean', f'{k}_std')] + ['n_runs']]
    summ.reset_index().to_csv(os.path.join(outdir, 'metrics_summary.csv'), index=False)
    print(summ[['f1_mean', 'f1_std', 'roc_auc_mean', 'roc_auc_std', 'n_runs']].round(4).to_string())
    cv = os.path.join(outdir, 'cv_fold_results.csv')
    if os.path.exists(cv):
        c = pd.read_csv(cv)
        cs = c.groupby('model')[cols].agg(['mean', 'std'])
        cs.columns = [f'{a}_{b}' for a, b in cs.columns]
        cs.reset_index().to_csv(os.path.join(outdir, 'cv_summary.csv'), index=False)
        print(cs[['f1_mean', 'f1_std', 'roc_auc_mean']].round(4).to_string())


if __name__ == '__main__':
    main()
