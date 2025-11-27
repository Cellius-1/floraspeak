import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.calibration import calibration_curve

def window_centers(N, win_s, hop_s): return np.arange(N)*hop_s + win_s/2.0

def load_embeddings(uid, emb_index):
    idx = pd.read_csv(emb_index)
    mask = idx["shard"].str.contains(f"uid{uid:04d}.npz", regex=False) | idx["embeddings"].str.contains(f"uid{uid:04d}_E.npz", regex=False)
    row = idx[mask]
    if row.empty: raise FileNotFoundError(f"uid{uid:04d} not found")
    return np.load(row.iloc[0]["embeddings"])["E"]

def build_labels(uid, N, win_s, hop_s, manifest_csv, horizon):
    man = pd.read_csv(manifest_csv)
    rec_dir = Path(man.loc[man["uid"] == uid, "path"].values[0])
    events = pd.read_csv(rec_dir/"events.csv")
    centers = window_centers(N, win_s, hop_s)
    y = np.zeros(N, dtype=int)
    for _, e in events.iterrows():
        t0 = float(e["onset_s"])
        y |= (centers >= (t0 - horizon)) & (centers < t0)
    return y

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0; total = len(y_true)
    for i in range(n_bins):
        m = (y_prob>=bins[i]) & (y_prob<bins[i+1])
        if m.sum()==0: continue
        conf = y_prob[m].mean()
        acc  = y_true[m].mean()
        ece += (m.sum()/total)*abs(acc-conf)
    return float(ece)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_index", default="results/ssl/embeddings_index.csv")
    ap.add_argument("--manifest",  default="data/synth/manifest.csv")
    ap.add_argument("--win_s", type=float, default=2.56)
    ap.add_argument("--hop_s", type=float, default=0.64)
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--alpha", type=float, default=0.05, help="target false-positive rate")
    ap.add_argument("--outdir", default="results/calibration")
    args = ap.parse_args()

    uids = pd.read_csv(args.manifest)["uid"].tolist()
    X_all, y_all, g_all = [], [], []
    for uid in uids:
        E = load_embeddings(uid, args.emb_index)
        y = build_labels(uid, E.shape[0], args.win_s, args.hop_s, args.manifest, args.horizon)
        if y.sum() < 5: continue
        X_all.append(E); y_all.append(y); g_all.append(np.full(len(y), uid))
    X = np.concatenate(X_all); y = np.concatenate(y_all); groups = np.concatenate(g_all)

    # split into train/cal/test by groups (uids)
    uniq = np.unique(groups)
    u_train, u_temp = train_test_split(uniq, test_size=0.4, random_state=0)
    u_cal,   u_test = train_test_split(u_temp, test_size=0.5, random_state=0)

    def mask(u): 
        m = np.isin(groups, u); 
        return X[m], y[m]

    Xtr, ytr = mask(u_train)
    Xcal, ycal = mask(u_cal)
    Xte,  yte  = mask(u_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, ytr)
    p_cal = clf.predict_proba(Xcal)[:,1]
    p_te  = clf.predict_proba(Xte )[:,1]

    # choose threshold to control FPR on calibration set
    # FPR = P(alert | y=0). We pick largest t s.t. FPR<=alpha
    neg = (ycal==0)
    cand = np.unique(np.round(p_cal,6))[::-1]
    thr = 0.5
    for t in cand:
        fpr = ( (p_cal[neg] >= t).sum() / max(1,neg.sum()) )
        if fpr <= args.alpha: thr = float(t); break

    # metrics
    auc  = roc_auc_score(yte, p_te)
    brier= brier_score_loss(yte, p_te)
    ece  = expected_calibration_error(yte, p_te, n_bins=10)
    conf = (p_te>=thr).mean()
    fpr  = ((p_te>=thr) & (yte==0)).sum() / max(1,(yte==0).sum())
    tpr  = ((p_te>=thr) & (yte==1)).sum() / max(1,(yte==1).sum())

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir)/f"calib_h{int(args.horizon)}.txt","w") as f:
        f.write(f"AUC={auc:.4f}\nBrier={brier:.4f}\nECE={ece:.4f}\nthr={thr:.4f}\nFPR@thr={fpr:.4f}\nTPR@thr={tpr:.4f}\nAlertRate={conf:.4f}\n")

    # reliability plot
    prob_true, prob_pred = calibration_curve(yte, p_te, n_bins=10, strategy="uniform")
    plt.figure(figsize=(4.2,3.6))
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Reliability (h={int(args.horizon)}s)  ECE={ece:.03f}")
    plt.tight_layout()
    plt.savefig(Path(args.outdir)/f"reliability_h{int(args.horizon)}.png", dpi=220)
    plt.close()

    # threshold–FPR tradeoff curve (calibration set)
    grid = np.linspace(0,1,101)
    fprs = [ ((p_cal[neg]>=t).sum()/max(1,neg.sum())) for t in grid ]
    plt.figure(figsize=(4.2,3.6))
    plt.plot(grid, fprs, "-")
    plt.axhline(args.alpha, color="k", linestyle="--", lw=1)
    plt.axvline(thr, color="k", linestyle=":", lw=1)
    plt.xlabel("Threshold t"); plt.ylabel("FPR on calibration")
    plt.title("False-alarm control")
    plt.tight_layout()
    plt.savefig(Path(args.outdir)/f"fpr_control_h{int(args.horizon)}.png", dpi=220)
    plt.close()

    print(f"[OK] AUC={auc:.3f}  ECE={ece:.3f}  thr={thr:.2f}  FPR={fpr:.3f}  TPR={tpr:.3f}")
