import math, numpy as np, pandas as pd
from pathlib import Path
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_index", default="results/vq_tokens/tokens_index.csv")
    ap.add_argument("--K", type=int, default=64)
    args = ap.parse_args()

    ti = pd.read_csv(args.tokens_index)
    if ti.empty:
        print("No rows in tokens_index.csv")
        raise SystemExit(1)

    all_tokens = []
    for _, r in ti.iterrows():
        arr = np.load(r["tokens"])
        all_tokens.append(arr.astype(int))

    a = np.concatenate(all_tokens)
    K = args.K
    hist = np.bincount(a, minlength=K)
    used = int((hist > 0).sum())
    p = hist / (hist.sum() + 1e-12)
    perp = float(math.exp(-((p[p > 0]) * np.log(p[p > 0])).sum()))

    print(f"Total tokens: {len(a)}")
    print(f"Used codes:  {used}/{K}")
    print(f"Perplexity:  {perp:.1f}")
    top = hist.argsort()[-20:][::-1]
    print("Top IDs:", top.tolist())
