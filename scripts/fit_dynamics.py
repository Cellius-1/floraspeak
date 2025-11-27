import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from sklearn.decomposition import PCA

def load_embeddings_for_uid(uid, emb_index_csv="results/ssl/embeddings_index.csv"):
    idx = pd.read_csv(emb_index_csv)
    # find row whose shard name matches uid####.npz
    uid_str = f"uid{uid:04d}.npz"
    row = idx[idx["shard"].str.endswith(uid_str)]
    if row.empty:
        # maybe embeddings live in results/ssl/emb/uid####_E.npz
        alt = idx[idx["embeddings"].str.contains(f"uid{uid:04d}_E.npz")]
        if alt.empty:
            raise FileNotFoundError(f"Embeddings for uid{uid:04d} not found in {emb_index_csv}")
        path = alt.iloc[0]["embeddings"]
    else:
        path = row.iloc[0]["embeddings"]
    E = np.load(path)["E"]  # [N_windows, d]
    return E

def window_centers(n, win_s, hop_s):
    return np.arange(n) * hop_s + win_s / 2.0

def pick_condition_indices(uid, condition, win_s, hop_s, manifest_csv="data/synth/manifest.csv"):
    man = pd.read_csv(manifest_csv)
    rec_path = man.loc[man["uid"] == uid, "path"].values[0]
    rec_dir = Path(rec_path)
    events = pd.read_csv(rec_dir / "events.csv")
    # windows time axis
    # infer N from embeddings later; here return callable
    return events[events["type"] == condition].copy()

def mask_windows_for_condition(N, cond_rows, win_s, hop_s):
    centers = window_centers(N, win_s, hop_s)
    keep = np.zeros(N, dtype=bool)
    for _, e in cond_rows.iterrows():
        t0 = float(e["onset_s"])
        t1 = t0 + float(e["duration_s"])
        keep |= (centers >= t0) & (centers <= t1)
    return keep

def edmd(X, r=8):
    """
    X: [T, d] sequence (time-ordered). Returns (evals, A_r, pca)
    """
    # PCA reduce to r
    pca = PCA(n_components=r, random_state=0)
    Z = pca.fit_transform(X)           # [T, r]
    Zm = Z[:-1].T                      # [r, T-1]
    Zp = Z[1:].T                       # [r, T-1]
    A = Zp @ pinv(Zm)                  # [r, r]
    evals = np.linalg.eigvals(A)
    return evals, A, pca

def plot_koopman(evals, title, outpath):
    fig, ax = plt.subplots(figsize=(5,5))
    th = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(th), np.sin(th), lw=1)                 # unit circle
    ax.scatter(evals.real, evals.imag, s=30)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Real"); ax.set_ylabel("Imag")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=220); plt.close(fig)

def try_sindy(Z, outpath_coef):
    try:
        import pysindy as ps
    except Exception:
        print("[SINDy] pysindy not installed; skipping.")
        return
    # Z: [T, r] reduced coordinates
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2),
                     optimizer=ps.STLSQ(alpha=0.01, threshold=0.05),
                     differentiation_method=ps.FiniteDifference())
    t = np.arange(Z.shape[0])  # dummy uniform dt
    model.fit(Z, t=t)
    # plot coefficient magnitudes
    coef = np.abs(model.coefficients())  # [n_features, r]
    plt.figure(figsize=(8,4))
    plt.imshow(coef.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="|coef|")
    plt.ylabel("State dim"); plt.xlabel("Feature")
    plt.title("SINDy coefficients (|·|)")
    plt.tight_layout(); plt.savefig(outpath_coef, dpi=220); plt.close()

def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for uid in args.uids:
        E = load_embeddings_for_uid(uid, args.emb_index)   # [N, d]
        N = E.shape[0]
        cond_rows = pick_condition_indices(uid, args.condition, args.win_s, args.hop_s, args.manifest)
        if cond_rows.empty:
            print(f"[{uid:04d}] no windows for condition {args.condition}")
            continue
        mask = mask_windows_for_condition(N, cond_rows, args.win_s, args.hop_s)
        X = E[mask]                                        # [T, d]
        if X.shape[0] < args.min_windows:
            print(f"[{uid:04d}] {args.condition} too short ({X.shape[0]} windows); skipping.")
            continue
        evals, A, pca = edmd(X, r=args.r)
        plot_koopman(evals, f"Koopman (uid{uid:04d}, {args.condition})", outdir / f"uid{uid:04d}_{args.condition}_koopman.png")

        # optional SINDy on reduced coords
        if args.sindy:
            Z = pca.transform(X)
            try_sindy(Z, outdir / f"uid{uid:04d}_{args.condition}_sindy_coeffs.png")
        print(f"[OK] uid{uid:04d} {args.condition}: T={X.shape[0]} windows -> plots saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_index", default="results/ssl/embeddings_index.csv")
    ap.add_argument("--manifest", default="data/synth/manifest.csv")
    ap.add_argument("--condition", default="drought")
    ap.add_argument("--uids", type=int, nargs="+", default=[2,0])
    ap.add_argument("--win_s", type=float, default=2.56)
    ap.add_argument("--hop_s", type=float, default=0.64)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--min_windows", type=int, default=50)
    ap.add_argument("--sindy", action="store_true")
    ap.add_argument("--outdir", default="results/poster_figures")
    args = ap.parse_args()
    main(args)