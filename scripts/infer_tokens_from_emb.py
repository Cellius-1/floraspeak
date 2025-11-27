import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from src.models.vq import VQVAE
import torch.nn.functional as F

def load_vq(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    m = VQVAE(d_in=ck["d_in"],
              d_hidden=ck["cfg"]["d_hidden"],
              d_code=ck["cfg"]["d_code"],
              n_codes=ck["cfg"]["n_codes"],
              beta=ck["cfg"]["beta"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vq, ck = load_vq(args.ckpt, device)

    shard_root = Path(args.shard_root)
    shard_paths = sorted(str(p) for p in shard_root.glob("uid*.npz"))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.no_grad():
        # normalized codebook for cosine NN
        cb = F.normalize(vq.vq.codebook, dim=1)

        for spath in shard_paths:
            E = np.load(spath)["E"]  # [N, d_in]
            X = torch.from_numpy(E).float().to(device)

            # encoder to code-space + L2 normalize
            z_e = vq.enc(X)
            z_e = F.normalize(z_e, dim=1)

            # cosine nearest neighbor (no EMA during inference)
            d = 2 - 2 * (z_e @ cb.t())      # [N, K]
            codes = torch.argmin(d, dim=1).cpu().numpy().astype(int)

            out_tokens = outdir / (Path(spath).stem + "_tokens.npy")
            np.save(out_tokens, codes)
            rows.append({"shard": spath, "tokens": str(out_tokens), "n": int(len(codes))})
            print(f"[TOK] {Path(spath).name} -> {out_tokens.name}  len={len(codes)}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(outdir/"tokens_index.csv", index=False)
    print(f"[DONE] {outdir/'tokens_index.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/vq/vq_final.pt")
    ap.add_argument("--shard_root", default="results/ssl/emb")
    ap.add_argument("--outdir", default="results/vq_tokens")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
