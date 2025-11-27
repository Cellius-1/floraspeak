import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def ensure(p): Path(p).mkdir(parents=True, exist_ok=True)

def parse_uid_from_name(stem: str) -> int:
    # handles "uid0000" or "uid0000_E"
    if not stem.startswith("uid"):
        raise ValueError(f"Unexpected shard name: {stem}")
    rest = stem[3:]             # after 'uid'
    rest = rest.split("_")[0]   # drop optional _E
    return int(rest)

def load_windows_for_uid(uid: int, shards_root="data/shards") -> np.ndarray:
    npz = Path(shards_root) / f"uid{uid:04d}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing window shard: {npz}")
    data = np.load(npz)
    # Expect X saved by make_shards.py
    if isinstance(data, np.lib.npyio.NpzFile):
        X = data["X"]
    else:
        X = data
    return X  # [N, W, C]

def fs_for_uid(uid: int, manifest: pd.DataFrame) -> float:
    rec_path = manifest.loc[manifest["uid"] == uid, "path"]
    if rec_path.empty:
        # fallback
        return 1000.0
    rec_dir = Path(rec_path.values[0])
    meta_path = rec_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return float(meta.get("fs", 1000.0))
    return 1000.0

def main(args):
    outdir = Path(args.outdir); ensure(outdir)
    tok_idx = pd.read_csv(args.tokens_index)
    man = pd.read_csv(args.manifest)

    # Build global token frequency to pick top-K
    freq = {}
    for _, r in tok_idx.iterrows():
        tokens = np.load(r["tokens"]).astype(int)  # [N]
        for t in tokens:
            freq[int(t)] = freq.get(int(t), 0) + 1

    top_tokens = [t for t, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:args.top_tokens]]
    print(f"Top tokens: {top_tokens}")

    # Collect examples per token across shards (cap per-shard)
    examples = {t: [] for t in top_tokens}
    uid_to_X_cache = {}
    uid_to_fs_cache = {}

    for _, r in tok_idx.iterrows():
        shard_path = Path(r["shard"])  # this points to .../uidXXXX.npz or .../uidXXXX_E.npz
        uid = parse_uid_from_name(shard_path.stem)

        # load windows for this uid from data/shards
        if uid not in uid_to_X_cache:
            uid_to_X_cache[uid] = load_windows_for_uid(uid, shards_root=args.shards_root)
            uid_to_fs_cache[uid] = fs_for_uid(uid, man)
        X = uid_to_X_cache[uid]              # [N, W, C]
        tokens = np.load(r["tokens"]).astype(int)  # [N]

        # safety: align lengths
        n = min(len(tokens), X.shape[0])
        tokens = tokens[:n]
        X = X[:n]

        # take up to M windows per token from this shard
        for t in top_tokens:
            idx = np.where(tokens == t)[0]
            if len(idx) == 0: continue
            take = idx[: max(1, args.examples_per_shard)]
            examples[t].append(X[take])   # list of [m, W, C]

    # Build and plot prototypes
    for t in top_tokens:
        if len(examples[t]) == 0: continue
        Xtok = np.concatenate(examples[t], axis=0)      # [K, W, C]
        # mean across channels first → [K, W]
        mean_per_win = Xtok.mean(axis=2)                # [K, W]
        mean_all = mean_per_win.mean(axis=0)            # [W]
        mean_ch  = Xtok.mean(axis=0)                    # [W, C]

        # PSD (quick FFT on mean waveform); fs from first uid seen (good enough for axis)
        fs_any = next(iter(uid_to_fs_cache.values()), args.fs)
        W = mean_all.shape[0]
        fft = np.fft.rfft(mean_all)
        psd = (fft * np.conj(fft)).real
        f = np.fft.rfftfreq(W, d=1/fs_any)

        # plot waveform + PSD
        fig, axes = plt.subplots(2, 1, figsize=(8,5))
        axes[0].plot(mean_all, lw=1)
        axes[0].set_title(f"Token {t} — prototype waveform (mean across channels, N={Xtok.shape[0]})")
        axes[0].set_xlabel("Samples"); axes[0].set_ylabel("Amplitude (z)")

        axes[1].plot(f, psd)
        axes[1].set_xlim(0, min(fs_any/2, 60))
        axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Power (a.u.)")
        axes[1].set_title("Prototype PSD")

        fig.tight_layout()
        fig.savefig(outdir / f"token_{t:03d}_prototype.png", dpi=200)
        plt.close(fig)

        # per-channel mean as a compact heatmap
        fig2, ax2 = plt.subplots(1,1, figsize=(8,3))
        im = ax2.imshow(mean_ch.T, aspect="auto", origin="lower", interpolation="nearest")
        ax2.set_title(f"Token {t} — per-channel mean waveform")
        ax2.set_xlabel("Samples"); ax2.set_ylabel("Channel")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        fig2.savefig(outdir / f"token_{t:03d}_per_channel.png", dpi=200)
        plt.close(fig2)

    print(f"[OK] wrote prototypes to {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_index", default="results/vq_tokens/tokens_index.csv")
    ap.add_argument("--manifest", default="data/synth/manifest.csv")
    ap.add_argument("--shards_root", default="data/shards")  # where uidXXXX.npz lives
    ap.add_argument("--outdir", default="results/poster_figures")
    ap.add_argument("--fs", type=float, default=1000)
    ap.add_argument("--top_tokens", type=int, default=12)
    ap.add_argument("--examples_per_shard", type=int, default=50)
    args = ap.parse_args()
    main(args)
