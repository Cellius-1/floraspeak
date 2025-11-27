import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os, math, json, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.ssl_masked import MaskedAutoencoderTS

class NPZWindows(Dataset):
    def __init__(self, index_csv, shard_root=None, limit=None):
        import pandas as pd
        self.df = pd.read_csv(index_csv)
        if limit: self.df = self.df.iloc[:limit].copy()
        if shard_root: self.df["shard"] = self.df["shard"].apply(lambda p: str(Path(shard_root)/Path(p).name))
        self.cache = {}  # lazy load per-shard
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        path = self.df.iloc[i]["shard"]
        if path not in self.cache:
            self.cache[path] = np.load(path)["X"]  # [N,W,C]
        X = self.cache[path]
        j = np.random.randint(0, X.shape[0])
        return X[j].astype(np.float32)  # [W,C]

def make_masks(B, L, mask_ratio=0.25, block=16):
    """ Create list of index tensors per batch item (blockwise masks). """
    idx_list = []
    M = max(1, int(L * mask_ratio))
    for _ in range(B):
        idx = []
        while len(idx) < M:
            start = np.random.randint(0, L)
            for k in range(block):
                t = start + k
                if t < L:
                    idx.append(t)
                if len(idx) >= M:
                    break
        idx = np.unique(np.array(idx, dtype=np.int64))
        idx_list.append(torch.from_numpy(idx))
    return idx_list

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = NPZWindows(args.index, shard_root=args.shard_root, limit=args.limit_shards)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    # infer channels/length from one sample
    x0 = ds[0]  # [W,C]
    L, C = x0.shape[0], x0.shape[1]
    model = MaskedAutoencoderTS(n_channels=C, d_model=args.d_model,
                                n_heads=args.heads, n_layers=args.layers,
                                p_drop=args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        for xb in dl:
            xb = xb.to(device)                         # [B,W,C]
            masks = make_masks(xb.size(0), xb.size(1), args.mask_ratio, args.mask_block)
            masks = [m.to(device) for m in masks]
            pred, targ, _ = model(xb, mask_idx=masks)  # [M,C], [M,C]
            loss = ((pred - targ)**2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch {epoch} step {global_step} | loss {loss.item():.5f}")

        # save each epoch
        outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
        ckpt = outdir / f"ssl_epoch{epoch:02d}.pt"
        torch.save({"model": model.state_dict(),
                    "cfg": vars(args),
                    "n_channels": C,
                    "win_len": L}, ckpt)
        print(f"[OK] saved {ckpt}")

    # save final symlink/copy
    final = outdir / "ssl_final.pt"
    torch.save({"model": model.state_dict(),
                "cfg": vars(args),
                "n_channels": C,
                "win_len": L}, final)
    print(f"[OK] saved {final}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/shards/index.csv")
    ap.add_argument("--shard_root", default="data/shards")
    ap.add_argument("--outdir", default="results/ssl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mask_ratio", type=float, default=0.25)
    ap.add_argument("--mask_block", type=int, default=16)
    ap.add_argument("--limit_shards", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    train(args)
