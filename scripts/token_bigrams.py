import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tok_idx = pd.read_csv(args.tokens_index)

    # Gather transitions across all shards
    counts = None
    n_codes = args.n_codes

    for _, r in tok_idx.iterrows():
        tok = np.load(r["tokens"]).astype(int)
        if len(tok) < 2: continue
        a = tok[:-1]; b = tok[1:]
        if counts is None:
            counts = np.zeros((n_codes, n_codes), dtype=np.int64)
        # clip codes into [0, n_codes)
        a = np.clip(a, 0, n_codes-1); b = np.clip(b, 0, n_codes-1)
        np.add.at(counts, (a, b), 1)

    if counts is None:
        print("No tokens found.")
        return

    # normalize to probabilities per row
    row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
    P = counts / row_sums

    # show top-k submatrix for readability (by marginal freq)
    marg = counts.sum(axis=1)
    top = np.argsort(-marg)[:args.top_tokens]
    Pk = P[np.ix_(top, top)]

    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(Pk, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(top))); ax.set_yticks(range(len(top)))
    ax.set_xticklabels(top, rotation=90); ax.set_yticklabels(top)
    ax.set_title("Token bigram probabilities (top-k tokens)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "token_bigrams_topk.png", dpi=220)
    plt.close(fig)

    print(f"[OK] wrote {outdir/'token_bigrams_topk.png'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_index", default="results/vq_tokens/tokens_index.csv")
    ap.add_argument("--outdir", default="results/poster_figures")
    ap.add_argument("--n_codes", type=int, default=256)
    ap.add_argument("--top_tokens", type=int, default=24)
    args = ap.parse_args()
    main(args)
