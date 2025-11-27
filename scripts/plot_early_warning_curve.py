import argparse
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/early_warning")
    ap.add_argument("--out", default="results/poster_figures/early_warning_curve.png")
    ap.add_argument("--horizons", type=int, nargs="+", default=[15,30,60,90,120])
    args = ap.parse_args()

    aucs, stds = [], []
    for h in args.horizons:
        p = Path(args.indir) / f"early_warning_h{h}.txt"
        if not p.exists(): continue
        mean,std = np.loadtxt(p, delimiter=",")
        aucs.append(mean); stds.append(std)
    xs = args.horizons[:len(aucs)]
    plt.figure(figsize=(5,3.5))
    plt.errorbar(xs, aucs, yerr=stds, fmt="-o")
    plt.xlabel("Forecast horizon (s before onset)")
    plt.ylabel("AUROC")
    plt.title("Early-warning performance")
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=220)
    plt.close()
    print(f"[PLOT] {args.out}")
