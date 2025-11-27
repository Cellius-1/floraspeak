import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--severities", type=str, nargs="+", default=["0.75","1.00","1.25","1.50"])
    ap.add_argument("--token_id", type=int, default=12)
    ap.add_argument("--out", default="results/poster_figures/dose_response.png")
    args = ap.parse_args()

    xs=[]; ys=[]
    for s in args.severities:
        tok_idx = Path(f"results/vq_tokens_sev_{s}")/"tokens_index.csv"
        if not tok_idx.exists(): continue
        ti = pd.read_csv(tok_idx)
        cnt=tot=0
        for _,r in ti.iterrows():
            a = np.load(r["tokens"]).astype(int)
            cnt += (a==args.token_id).sum()
            tot += len(a)
        if tot>0:
            xs.append(float(s)); ys.append(cnt/tot)
    order = np.argsort(xs); xs=np.array(xs)[order]; ys=np.array(ys)[order]
    plt.figure(figsize=(4.2,3.2))
    plt.plot(xs, ys, "-o")
    plt.xlabel("Drought severity (scale factor)"); plt.ylabel(f"Frequency of token {args.token_id}")
    plt.title("Counterfactual dose–response")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=220); plt.close()
    print(f"[PLOT] {args.out}")
