import argparse
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, networkx as nx

def plot_adj(A, title, outpath):
    C = A.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(C))
    for i in range(C):
        for j in range(C):
            if i!=j and A[i,j] > 0:
                G.add_edge(i, j, weight=A[i,j])
    # layout and draw
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_labels(G, pos, labels={i:str(i) for i in range(C)}, font_size=10)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=12, width=1.5, connectionstyle='arc3,rad=0.1')
    plt.title(title)
    plt.axis('off')
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"[PLOT] {outpath}")

def main(args):
    indir = Path(args.indir)
    # pick a couple of sessions + conditions for the poster
    for uid in args.uids:
        for cond in args.conditions:
            A_path = indir / f"uid{uid:04d}_{cond}_adj.npy"
            if not A_path.exists(): 
                continue
            A = np.load(A_path)
            plot_adj(A, f"Routing (Granger+FDR) — uid{uid:04d} | {cond}", 
                     Path(args.outdir) / f"uid{uid:04d}_{cond}_graph.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/causal")
    ap.add_argument("--outdir", default="results/poster_figures")
    ap.add_argument("--uids", type=int, nargs="+", default=[2,0])  # choose rich ones
    ap.add_argument("--conditions", nargs="+", default=["rest","drought","light","heat"])
    args = ap.parse_args()
    main(args)
