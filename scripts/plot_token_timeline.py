import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def main(args):
    tokens_path = Path(args.tokens)
    # infer uid and recording dir from manifest
    uid = int(tokens_path.stem.replace("uid","").split("_")[0])
    man = pd.read_csv(args.manifest)
    rec_row = man.loc[man["uid"] == uid]
    if rec_row.empty:
        raise FileNotFoundError(f"uid{uid:04d} not found in manifest.")
    rec_dir = Path(rec_row["path"].values[0])

    tokens = np.load(tokens_path).astype(int)   # [N_windows]
    # timing using shard params (win_s, hop_s) assumed fixed in your pipeline
    win_s = args.win_s
    hop_s = args.hop_s
    centers = np.arange(tokens.shape[0]) * hop_s + win_s/2.0

    # events
    events = pd.read_csv(rec_dir / "events.csv")
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(centers, tokens, s=8, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Token ID")
    ax.set_title(f"Token timeline — uid{uid:04d}")

    # event overlays
    for _, e in events.iterrows():
        t0 = float(e["onset_s"]); dur = float(e["duration_s"])
        lab = str(e["type"])
        ax.axvspan(t0, t0+dur, alpha=0.15, label=lab)
    # avoid duplicate labels in legend
    handles, labels = [], []
    for coll in ax.collections:
        pass
    seen = set()
    for patch in ax.patches:
        lab = patch.get_label()
        if lab not in seen:
            handles.append(patch); labels.append(lab); seen.add(lab)
    if handles:
        ax.legend(handles, labels, loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="e.g., results/vq_tokens/uid0002_tokens.npy")
    ap.add_argument("--manifest", default="data/synth/manifest.csv")
    ap.add_argument("--win_s", type=float, default=2.56)
    ap.add_argument("--hop_s", type=float, default=0.64)
    ap.add_argument("--out", default="results/poster_figures/token_timeline.png")
    args = ap.parse_args()
    main(args)
