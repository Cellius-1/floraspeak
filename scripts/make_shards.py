# scripts/make_shards.py
# Create fixed-size window shards from sessions listed in a manifest.
# Robust to CSV format (with/without 'time' column, header/no-header).
# Ensures channel count == 8 to match the SSL model.

import argparse
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend


def harmonize_channels(sig: np.ndarray, n_channels: int = 8) -> np.ndarray:
    """
    Ensure a fixed channel count for downstream models.
    - If more channels than n_channels, drop extras.
    - If fewer, zero-pad to the right.
    """
    if sig.ndim != 2:
        raise ValueError(f"signals array must be 2D [T, C], got shape {sig.shape}")
    T, C = sig.shape
    if C == n_channels:
        return sig
    if C > n_channels:
        return sig[:, :n_channels]
    pad = np.zeros((T, n_channels - C), dtype=sig.dtype)
    return np.hstack([sig, pad])


def load_signals_csv(sig_path: Path) -> np.ndarray:
    """
    Robust CSV loader:
      - If header exists and includes 'time', drop it.
      - If header exists without 'time', use all numeric columns.
      - If no header (np.savetxt-like), read as numeric-only.
    Returns float array [T, C].
    """
    try:
        df = pd.read_csv(sig_path)
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        sig = df.to_numpy(dtype=float)
    except Exception:
        # fallback: no header numeric CSV
        df = pd.read_csv(sig_path, header=None)
        sig = df.to_numpy(dtype=float)
    return sig


def butter_bandpass(lo, hi, fs, order=4):
    nyq = 0.5 * fs
    lo_n = lo / nyq if lo is not None else None
    hi_n = hi / nyq if hi is not None else None
    if lo_n and hi_n:
        b, a = butter(order, [lo_n, hi_n], btype='bandpass')
    elif lo_n:
        b, a = butter(order, lo_n, btype='highpass')
    elif hi_n:
        b, a = butter(order, hi_n, btype='lowpass')
    else:
        raise ValueError("At least one of lo/hi must be specified.")
    return b, a


def preprocess(sig: np.ndarray, fs: float,
               hp_hz: float = 0.1, lp_hz: float = 40.0,
               order: int = 4) -> np.ndarray:
    """
    Simple, stable preprocessing:
      1) Linear detrend per channel
      2) Butterworth bandpass/highpass (filtfilt) per channel
    """
    # Detrend (remove linear trend)
    x = detrend(sig, axis=0, type='linear')
    # Band-pass (or high-pass if lp_hz is None)
    b, a = butter_bandpass(hp_hz, lp_hz, fs, order=order)
    # Apply per channel with filtfilt (zero-phase)
    for c in range(x.shape[1]):
        x[:, c] = filtfilt(b, a, x[:, c], method="pad")
    return x


def window_signal(sig: np.ndarray, fs: float, win_s: float, hop_s: float) -> np.ndarray:
    """
    Slice the continuous signal [T, C] into overlapping windows:
      X -> [N_windows, W, C]
    """
    T, C = sig.shape
    W = int(round(win_s * fs))
    H = int(round(hop_s * fs))
    if W <= 0 or H <= 0:
        raise ValueError(f"Invalid W/H with fs={fs}, win_s={win_s}, hop_s={hop_s}")
    if T < W:
        return np.zeros((0, W, C), dtype=sig.dtype)
    n = 1 + (T - W) // H
    X = np.empty((n, W, C), dtype=sig.dtype)
    for i in range(n):
        s = i * H
        X[i] = sig[s:s+W, :]
    return X


def process_recording(rec_dir: Path, fs: float, win_s: float, hop_s: float,
                      n_channels: int = 8) -> np.ndarray:
    """
    Load one session, harmonize channels, preprocess, and window.
    """
    sig_path = rec_dir / "signals.csv"
    if not sig_path.exists():
        raise FileNotFoundError(f"Missing signals.csv at {sig_path}")
    sig = load_signals_csv(sig_path)              # [T, C?]
    sig = harmonize_channels(sig, n_channels=n_channels)
    x = preprocess(sig, fs=fs)                    # [T, C]
    X = window_signal(x, fs=fs, win_s=win_s, hop_s=hop_s)  # [N, W, C]
    return X


def main():
    ap = argparse.ArgumentParser(description="Shard plant recordings into fixed windows.")
    ap.add_argument("--manifest", required=True, help="CSV with columns: uid,path")
    ap.add_argument("--outdir", required=True, help="Output folder for uidXXXX.npz shards")
    ap.add_argument("--win_s", type=float, default=2.56)
    ap.add_argument("--hop_s", type=float, default=0.64)
    ap.add_argument("--n_channels", type=int, default=8, help="Force channel count to this value")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for _, row in manifest.iterrows():
        uid = int(row["uid"])
        rec_dir = Path(row["path"])
        meta_path = rec_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta.json in {rec_dir}")
        with open(meta_path) as f:
            meta = json.load(f)
        fs = float(meta.get("fs", 1000.0))

        X = process_recording(rec_dir, fs, args.win_s, args.hop_s, n_channels=args.n_channels)
        shard_path = outdir / f"uid{uid:04d}.npz"
        np.savez(shard_path, X=X)

        index_rows.append({
            "uid": uid,
            "shard": str(shard_path),
            "n_windows": int(X.shape[0]),
            "win_s": args.win_s,
            "hop_s": args.hop_s,
            "fs": fs,
            "n_channels": args.n_channels,
        })
        print(f"[OK] uid{uid:04d}: X shape={X.shape} -> {shard_path}")

    idx_df = pd.DataFrame(index_rows)
    idx_csv = outdir / "index.csv"
    idx_df.to_csv(idx_csv, index=False)
    print(f"[DONE] wrote shard index -> {idx_csv}")


if __name__ == "__main__":
    main()
