import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx

def load_adj(indir, condition):
    mats = []
    for p in Path(indir).glob(f"uid*_{condition}_adj.npy"):
        mats.append(np.load(p))
    return mats

def consensus(A_list, tau=0.6):
    # frequency matrix (fraction of sessions that contain the edge)
    if not A_list: return None, None
    S = len(A_list); C = A_list[0].shape[0]
    F = np.zeros((C,C), float)
    for A in A_list: F += (A>0).astype(float)
    F /= S
    inv = (F >= tau).astype(float)
    return F, inv

def plot_heat(F, title, out):
    plt.figure(figsize=(4.2,3.8))
    plt.imshow(F, vmin=0, vmax=1, cmap="viridis", origin="lower")
    plt.colorbar(label="edge frequency")
    plt.title(title); plt.xlabel("to j"); plt.ylabel("from i")
    plt.tight_layout(); Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220); plt.close(); print(f"[PLOT] {out}")

def plot_delta(invA, invB, title, out):
    # edges present in B (stress) but not in A (rest)
    D = (invB>0) & (invA==0)
    C = D.shape[0]
    G = nx.DiGraph()
    for i in range(C): G.add_node(i)
    for i in range(C):
        for j in range(C):
            if D[i,j]:
                G.add_edge(i,j)
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(5,5))
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos, labels={i:str(i) for i in range(C)}, font_size=10)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=12, width=1.6, connectionstyle='arc3,rad=0.1')
    plt.title(title); plt.axis("off"); plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=240); plt.close(); print(f"[PLOT] {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/causal")
    ap.add_argument("--condA", default="rest")
    ap.add_argument("--condB", default="drought")
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--outdir", default="results/poster_figures")
    args = ap.parse_args()

    A_list = load_adj(args.indir, args.condA)
    B_list = load_adj(args.indir, args.condB)
    FA, invA = consensus(A_list, args.tau)
    FB, invB = consensus(B_list, args.tau)

    if FA is None or FB is None:
        raise SystemExit("No adjacency matrices found. Run build_causal_graphs first.")

    plot_heat(FA, f"Invariant frequency — {args.condA} (τ={args.tau})", Path(args.outdir)/f"invariants_{args.condA}.png")
    plot_heat(FB, f"Invariant frequency — {args.condB} (τ={args.tau})", Path(args.outdir)/f"invariants_{args.condB}.png")
    plot_delta(invA, invB, f"Stress-specific edges ({args.condB} minus {args.condA})", Path(args.outdir)/f"delta_graph_{args.condA}_to_{args.condB}.png")
