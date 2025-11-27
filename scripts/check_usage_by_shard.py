import numpy as np, pandas as pd, argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_index", default="results/vq_tokens/tokens_index.csv")
    ap.add_argument("--K", type=int, default=16)
    args = ap.parse_args()
    ti = pd.read_csv(args.tokens_index)
    rows=[]
    for _,r in ti.iterrows():
        tok = np.load(r["tokens"]).astype(int)
        hist = np.bincount(tok, minlength=args.K)
        used = int((hist>0).sum())
        rows.append({"shard": r["shard"], "used": used})
    df = pd.DataFrame(rows).sort_values("used")
    print(df.to_string(index=False))
