import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import json

def label_windows(rec_dir: Path, n_windows: int, fs: int, win_s: float, hop_s: float):
    events = pd.read_csv(rec_dir / "events.csv")
    centers = np.arange(n_windows) * hop_s + win_s / 2.0
    labels = np.array(["rest"] * n_windows, dtype=object)
    for _, e in events.iterrows():
        onset = float(e["onset_s"]); dur = float(e["duration_s"]); typ = str(e["type"])
        mask = (centers >= onset) & (centers <= onset + dur)
        labels[mask] = typ
    return labels

def shard_uid_from_path(p: str | Path) -> int:
    # expects names like uid0007.npz or uid0007_E.npz
    stem = Path(p).stem
    if "_E" in stem:
        stem = stem.split("_E")[0]
    if not stem.startswith("uid"):
        raise ValueError(f"Cannot parse uid from {p}")
    return int(stem.replace("uid", ""))

def main(args):
    tok_idx = pd.read_csv(args.tokens_index)
    emb_idx = pd.read_csv(args.emb_index)
    manifest = pd.read_csv(args.manifest)

    # Build uid -> recording path map from manifest
    uid_to_recpath = {int(r["uid"]): r["path"] for _, r in manifest.iterrows()}

    rows = []
    rng = np.random.default_rng(42)

    for _, r in tok_idx.iterrows():
        uid = shard_uid_from_path(r["shard"])
        rec_path = uid_to_recpath.get(uid, None)
        if rec_path is None or not Path(rec_path).exists():
            raise FileNotFoundError(f"No recording path for uid{uid:04d}. "
                                    f"Check {args.manifest} and your shards/tokens files.")
        rec_dir = Path(rec_path)

        # meta + timing
        meta = json.load(open(rec_dir / "meta.json"))
        fs = int(meta["fs"])
        n = int(r["n"])

        # labels per window
        labels = label_windows(rec_dir, n_windows=n, fs=fs, win_s=args.win_s, hop_s=args.hop_s)

        # tokens for this shard
        tokens = np.load(r["tokens"])

        # MI and shuffle control
        mi = mutual_info_score(tokens, labels)
        shuf = [mutual_info_score(rng.permutation(tokens), labels) for _ in range(args.n_shuffles)]
        shuf = np.array(shuf, float)
        z = float((mi - shuf.mean()) / (shuf.std() + 1e-12))

        rows.append({
            "uid": uid,
            "recording_dir": str(rec_dir),
            "mi": float(mi),
            "shuffle_mean": float(shuf.mean()),
            "shuffle_std": float(shuf.std()),
            "z_score": z,
            "n_windows": n
        })
        print(f"[MI] uid{uid:04d}: MI={mi:.5f}  z={z:.2f} vs shuffle")

    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[DONE] wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_index", default="results/vq_tokens/tokens_index.csv")
    ap.add_argument("--emb_index", default="results/ssl/embeddings_index.csv")  # kept for future use
    ap.add_argument("--manifest", default="data/synth/manifest.csv")
    ap.add_argument("--win_s", type=float, default=2.56)
    ap.add_argument("--hop_s", type=float, default=0.64)
    ap.add_argument("--n_shuffles", type=int, default=1000)
    ap.add_argument("--out", default="results/mi_tokens.csv")
    args = ap.parse_args()
    main(args)
