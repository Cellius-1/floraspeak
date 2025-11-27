# scripts/build_causal_graphs.py
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import decimate
from statsmodels.tsa.stattools import grangercausalitytests

# Benjamini–Hochberg FDR
def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m+1) / m)
    passed = p[order] <= thresh
    k = np.where(passed)[0].max()+1 if passed.any() else 0
    cutoff = p[order][k-1] if k>0 else 0.0
    return (p <= cutoff) if k>0 else np.zeros_like(p, dtype=bool)

def merge_intervals(itvs):
    """Merge [start,end) index intervals (in samples)."""
    if not itvs: return []
    itvs = sorted(itvs)
    out = [list(itvs[0])]
    for s,e in itvs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s,e])
    return [(s,e) for s,e in out]

def complement_intervals(itvs, T):
    """Complement of merged intervals within [0,T)."""
    itvs = merge_intervals(itvs)
    out = []
    cur = 0
    for s,e in itvs:
        if cur < s:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < T:
        out.append((cur, T))
    return out

def segment_by_condition(sig, fs_in, events_df, condition, ds_factor=100, min_seconds=15.0):
    """
    Returns (Xd, fs_out) for a given condition.
    - If condition == 'rest': take complement of all event intervals.
    - Else: concatenate intervals where events['type'] == condition.
    Downsample by ds_factor (zero-phase).
    """
    T = sig.shape[0]
    # Build event intervals in sample indices
    ev_itvs = []
    for _, e in events_df.iterrows():
        t0 = float(e["onset_s"]); dur = float(e["duration_s"])
        s = max(0, int(round(t0 * fs_in)))
        eidx = min(T, int(round((t0 + dur) * fs_in)))
        if eidx > s:
            ev_itvs.append((s, eidx))

    if condition == "rest":
        itvs = complement_intervals(ev_itvs, T)
    else:
        itvs = []
        for _, e in events_df.iterrows():
            if str(e["type"]) == condition:
                t0 = float(e["onset_s"]); dur = float(e["duration_s"])
                s = max(0, int(round(t0 * fs_in)))
                eidx = min(T, int(round((t0 + dur) * fs_in)))
                if eidx > s:
                    itvs.append((s, eidx))
        itvs = merge_intervals(itvs)

    # drop too-short intervals BEFORE concat/decimate
    min_len = int(min_seconds * fs_in)
    itvs = [(s,e) for s,e in itvs if (e - s) >= min_len]
    if not itvs:
        return None, None

    X = np.concatenate([sig[s:e] for s,e in itvs], axis=0)
    if ds_factor > 1:
        Xd = decimate(X, ds_factor, axis=0, zero_phase=True)
        fs_out = fs_in / ds_factor
    else:
        Xd, fs_out = X, fs_in
    return Xd, fs_out

def compute_granger_matrix(X, maxlag=2):
    """Pairwise Granger p-values P[i,j] for i→j."""
    T, C = X.shape
    P = np.ones((C, C), float)
    for j in range(C):
        for i in range(C):
            if i == j: 
                continue
            data = np.column_stack([X[:, j], X[:, i]])  # target, source
            try:
                res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                pmin = min(res[lag][0]["ssr_ftest"][1] for lag in res)
                P[i, j] = pmin
            except Exception:
                P[i, j] = 1.0
    return P

def main(args):
    man = pd.read_csv(args.manifest)
    # If you want to restrict, edit this list; 'rest' now supported by complement logic.
    default_conditions = ["rest","light","touch","heat","cold","nutrient","drought"]
    conditions = args.conditions if args.conditions else default_conditions

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for _, row in man.iterrows():
        rec_dir = Path(row["path"]); uid = int(row["uid"])
        sig = pd.read_csv(rec_dir/"signals.csv").values  # [T,C]
        with open(rec_dir/"meta.json") as f:
            meta = json.load(f)
        fs = float(meta["fs"])
        events = pd.read_csv(rec_dir/"events.csv")

        for cond in conditions:
            Xc, fs_c = segment_by_condition(sig, fs, events, cond,
                                            ds_factor=args.ds_factor,
                                            min_seconds=args.min_seconds)
            if Xc is None:
                # nothing long enough for this condition
                continue
            # z-score per channel
            Xc = (Xc - Xc.mean(axis=0, keepdims=True)) / (Xc.std(axis=0, keepdims=True) + 1e-9)
            P = compute_granger_matrix(Xc, maxlag=args.maxlag)  # p-values
            C = P.shape[0]
            A = np.zeros_like(P, dtype=float)
            # FDR per target
            for j in range(C):
                pv = P[:, j].copy()
                pv[j] = 1.0
                mask = np.ones(C, dtype=bool); mask[j] = False
                sigmask = bh_fdr(pv[mask], alpha=args.alpha)
                idxs = np.where(mask)[0]
                A[idxs[sigmask], j] = 1.0

            np.save(outdir / f"uid{uid:04d}_{cond}_adj.npy", A)
            np.save(outdir / f"uid{uid:04d}_{cond}_pvals.npy", P)
            print(f"[OK] uid{uid:04d} {cond}: edges={int(A.sum())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/synth/manifest.csv")
    ap.add_argument("--outdir", default="results/causal")
    ap.add_argument("--ds_factor", type=int, default=100)
    ap.add_argument("--maxlag", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min_seconds", type=float, default=15.0)
    ap.add_argument("--conditions", nargs="*", help="Override conditions to compute (e.g., rest drought heat)")
    args = ap.parse_args()
    main(args)
