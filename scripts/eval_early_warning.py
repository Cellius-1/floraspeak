import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

def load_embeddings(uid, emb_index):
    idx = pd.read_csv(emb_index)
    # match by uid#### in either 'shard' or 'embeddings'
    mask = idx["shard"].str.contains(f"uid{uid:04d}.npz", regex=False) | idx["embeddings"].str.contains(f"uid{uid:04d}_E.npz", regex=False)
    row = idx[mask]
    if row.empty: raise FileNotFoundError(f"uid{uid:04d} not found in {emb_index}")
    return np.load(row.iloc[0]["embeddings"])["E"]  # [N, d]

def window_centers(N, win_s, hop_s):
    return np.arange(N) * hop_s + win_s / 2.0

def build_labels(uid, N, win_s, hop_s, manifest_csv, horizon_s):
    man = pd.read_csv(manifest_csv)
    rec_dir = Path(man.loc[man["uid"] == uid, "path"].values[0])
    events = pd.read_csv(rec_dir/"events.csv")
    centers = window_centers(N, win_s, hop_s)
    y = np.zeros(N, dtype=int)
    # mark positive if center in [onset - horizon, onset)
    for _, e in events.iterrows():
        t0 = float(e["onset_s"])
        y |= (centers >= (t0 - horizon_s)) & (centers < t0)
    return y

def main(args):
    uids = pd.read_csv(args.manifest)["uid"].tolist()
    X_all, y_all, g_all = [], [], []
    for uid in uids:
        E = load_embeddings(uid, args.emb_index)
        y = build_labels(uid, E.shape[0], args.win_s, args.hop_s, args.manifest, args.horizon)
        if y.sum() < 5:  # not enough positives
            continue
        X_all.append(E); y_all.append(y); g_all.append(np.full(len(y), uid))
    X = np.concatenate(X_all); y = np.concatenate(y_all); groups = np.concatenate(g_all)

    # class balance print
    pos = int(y.sum()); neg = int((1-y).sum())
    print(f"[data] N={len(y)}  pos={pos} neg={neg}  pos_rate={pos/len(y):.3f}")

    # group K-fold CV by uid
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:,1]
        auc = roc_auc_score(y[te], p)
        aucs.append(auc)
    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
    print(f"[AUROC] horizon={args.horizon}s  mean={auc_mean:.3f}  std={auc_std:.3f}")

    # save a point for plotting across horizons
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"early_warning_h{int(args.horizon)}.txt", "w") as f:
        f.write(f"{auc_mean:.6f},{auc_std:.6f}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_index", default="results/ssl/embeddings_index.csv")
    ap.add_argument("--manifest",  default="data/synth/manifest.csv")
    ap.add_argument("--win_s", type=float, default=2.56)
    ap.add_argument("--hop_s", type=float, default=0.64)
    ap.add_argument("--horizon", type=float, default=60.0, help="seconds before onset = positive")
    ap.add_argument("--outdir", default="results/early_warning")
    args = ap.parse_args()
    main(args)
