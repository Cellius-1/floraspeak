import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from src.models.ssl_masked import MaskedAutoencoderTS

class OneShard(Dataset):
    def __init__(self, shard_path):
        self.X = np.load(shard_path)["X"]  # [N,W,C]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i].astype(np.float32)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    ckpt = torch.load(args.ckpt, map_location=device)
    n_channels = ckpt["n_channels"]
    cfg = ckpt["cfg"]
    model = MaskedAutoencoderTS(
        n_channels=n_channels,
        d_model=cfg["d_model"],
        n_heads=cfg["heads"],
        n_layers=cfg["layers"],
        p_drop=cfg["drop"]
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    embdir = outdir / "emb"; embdir.mkdir(parents=True, exist_ok=True)

    shard_root = Path(args.shard_root)
    shard_paths = sorted(shard_root.glob("uid*.npz"))

    index_rows = []
    with torch.no_grad():
        for spath in shard_paths:
            ds = OneShard(spath)
            dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
            chunk = []
            for xb in dl:
                xb = xb.to(device)
                z = model.embed(xb).cpu().numpy()  # [B, d_model]
                chunk.append(z)
            E = np.concatenate(chunk, axis=0) if chunk else np.zeros((0, cfg["d_model"]), dtype=np.float32)

            out_path = embdir / (spath.stem + "_E.npz")
            np.savez_compressed(out_path, E=E)

            index_rows.append({
                "shard": str(spath),
                "embeddings": str(out_path),
                "n_windows": int(E.shape[0]),
                "d_model": int(E.shape[1]) if E.size else int(cfg["d_model"])
            })
            print(f"[OK] {spath.name} -> {out_path.name}  shape={E.shape}")

    # write small index
    import pandas as pd
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_csv(outdir / "embeddings_index.csv", index=False)
    print(f"[DONE] Wrote {outdir/'embeddings_index.csv'} with {len(index_rows)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ssl/ssl_final.pt")
    ap.add_argument("--shard_root", default="data/shards")
    ap.add_argument("--outdir", default="results/ssl")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    main(args)
