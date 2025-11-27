import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def load_pairs(emb_index, manifest, win_s, hop_s, horizon):
    idx = pd.read_csv(emb_index); man = pd.read_csv(manifest)
    rows=[]
    for _, r in man.iterrows():
        uid = int(r["uid"])
        m = idx["shard"].str.contains(f"uid{uid:04d}.npz", regex=False) | idx["embeddings"].str.contains(f"uid{uid:04d}_E.npz", regex=False)
        row = idx[m]
        if row.empty: continue
        E = np.load(row.iloc[0]["embeddings"])["E"]
        # labels
        events = pd.read_csv(Path(r["path"])/"events.csv")
        centers = np.arange(E.shape[0])*hop_s + win_s/2.0
        y = np.zeros(E.shape[0], dtype=int)
        for _, e in events.iterrows():
            t0 = float(e["onset_s"])
            y |= (centers >= (t0 - horizon)) & (centers < t0)
        rows.append((E,y,uid))
    X = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    g = np.concatenate([np.full(len(r[1]), r[2]) for r in rows])
    return X,y,g

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_index_A", default="results/ssl_domA/embeddings_index.csv")
    ap.add_argument("--manifest_A",  default="data/synth_domains/A/manifest.csv")
    ap.add_argument("--emb_index_B", default="results/ssl_domB/embeddings_index.csv")
    ap.add_argument("--manifest_B",  default="data/synth_domains/B/manifest.csv")
    ap.add_argument("--win_s", type=float, default=2.56)
    ap.add_argument("--hop_s", type=float, default=0.64)
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--out_csv", default="results/domain_eval/auroc_table.csv")
    args = ap.parse_args()

    Path("results/domain_eval").mkdir(parents=True, exist_ok=True)

    XA, yA, gA = load_pairs(args.emb_index_A, args.manifest_A, args.win_s, args.hop_s, args.horizon)
    XB, yB, gB = load_pairs(args.emb_index_B, args.manifest_B, args.win_s, args.hop_s, args.horizon)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")

    # Train on A, test on A (in-domain) and B (cross-domain)
    clf.fit(XA, yA)
    auc_AA = roc_auc_score(yA, clf.predict_proba(XA)[:,1])
    auc_AB = roc_auc_score(yB, clf.predict_proba(XB)[:,1])

    # Train on B, test on B and A
    clf.fit(XB, yB)
    auc_BB = roc_auc_score(yB, clf.predict_proba(XB)[:,1])
    auc_BA = roc_auc_score(yA, clf.predict_proba(XA)[:,1])

    df = pd.DataFrame([
        {"Train":"A","Test":"A","AUROC":auc_AA},
        {"Train":"A","Test":"B","AUROC":auc_AB},
        {"Train":"B","Test":"B","AUROC":auc_BB},
        {"Train":"B","Test":"A","AUROC":auc_BA},
    ])
    df.to_csv(args.out_csv, index=False)
    print(df)
    print(f"[OK] wrote {args.out_csv}")
