import argparse, json, math, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.models.vq import VQVAE

# Optional: reduce OpenMP thread contention on Windows (quiet the warning)
os.environ.setdefault("OMP_NUM_THREADS", "1")

class EmbDS(Dataset):
    def __init__(self, emb_index_csv):
        import pandas as pd
        self.df = pd.read_csv(emb_index_csv)
        self.paths = self.df["embeddings"].tolist()
        self.cache = {}
        self.index = []
        for p in self.paths:
            E = np.load(p)["E"]  # [N, d_in]
            self.index += [(p, i) for i in range(E.shape[0])]
    def __len__(self): return len(self.index)
    def __getitem__(self, k):
        p, i = self.index[k]
        if p not in self.cache: self.cache[p] = np.load(p)["E"]
        return self.cache[p][i].astype(np.float32)  # [d_in]

def kmeans_init(model: VQVAE, ds: EmbDS, n_codes: int, sample_cap=100_000, seed=0, device="cpu"):
    """
    K-means initialize the VQ codebook in the *code space* (after encoder).
    Ensures centroid shape matches d_code.
    """
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)

    # sample embeddings
    S = min(sample_cap, len(ds))
    idxs = rng.integers(0, len(ds), size=S)
    Xs = torch.from_numpy(np.stack([ds[i] for i in idxs], axis=0)).float().to(device)  # [S, d_in]

    # project to code space (d_code) and L2-normalize
    with torch.no_grad():
        Z = model.enc(Xs)                # [S, d_code]
        Z = F.normalize(Z, dim=1)
        Z_np = Z.cpu().numpy().astype("float32")

    km = MiniBatchKMeans(n_clusters=n_codes, batch_size=2048, n_init="auto", random_state=seed).fit(Z_np)
    centroids = km.cluster_centers_.astype("float32")  # [K, d_code]

    with torch.no_grad():
        cb = torch.from_numpy(centroids).to(device)
        cb = F.normalize(cb, dim=1)
        model.vq.codebook.copy_(cb)
        model.vq.ema_w.copy_(cb)
        model.vq.ema_cluster_size.fill_(Z_np.shape[0] / n_codes)

    print(f"[init] KMeans on code-space: {n_codes}x{centroids.shape[1]} from {Z_np.shape[0]} samples.")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # dataset & loader
    ds = EmbDS(args.emb_index)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    # infer d_in
    d_in = ds[0].shape[0]
    model = VQVAE(d_in=d_in, d_hidden=args.d_hidden, d_code=args.d_code,
                  n_codes=args.n_codes, beta=args.beta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # K-means codebook initialization (in code space)
    kmeans_init(model, ds, n_codes=args.n_codes, seed=0, device=device)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for xb in dl:
            xb = xb.to(device)  # [B, d_in]
            # forward
            _, codes, loss, logs = model(xb)

            # usage-entropy regularizer (encourage spread across codes)
            p = torch.bincount(codes, minlength=args.n_codes).float().to(device)
            p = p / (p.sum() + 1e-12)
            ent = -(p[p > 0] * p[p > 0].log()).sum()
            uniform_ent = math.log(args.n_codes)
            usage_loss = args.entropy_w * (uniform_ent - ent)
            loss = loss + usage_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                used = int((p > 0).sum().item())
                perplexity = float(torch.exp(-(p[p>0]*p[p>0].log()).sum()).item()) if used > 0 else 0.0
                print(f"ep{epoch} step{step} "
                      f"loss={loss.item():.5f} rec={logs['rec'].item():.5f} vq={logs['vq'].item():.5f} "
                      f"used={used}/{args.n_codes} perp={perplexity:.1f}")

            # Dead-code revival every N steps
            if step % 500 == 0:
                with torch.no_grad():
                    dead = (model.vq.ema_cluster_size < 1.0).nonzero(as_tuple=True)[0]
                    if len(dead) > 0:
                        z_e = model.enc(xb)
                        z_e = F.normalize(z_e, dim=1)
                        take = z_e[torch.randint(0, z_e.size(0), (len(dead),), device=device)]
                        model.vq.codebook[dead] = take
                        model.vq.ema_w[dead] = take
                        model.vq.ema_cluster_size[dead] = 10.0
                        print(f"[revive] reinit {len(dead)} codes")

        # save each epoch
        ckpt = outdir / f"vq_epoch{epoch:02d}.pt"
        torch.save({"model": model.state_dict(), "cfg": vars(args), "d_in": d_in}, ckpt)
        print(f"[OK] saved {ckpt}")

    final = outdir / "vq_final.pt"
    torch.save({"model": model.state_dict(), "cfg": vars(args), "d_in": d_in}, final)
    print(f"[OK] saved {final}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_index", default="results/ssl/embeddings_index.csv")
    ap.add_argument("--outdir", default="results/vq")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d_hidden", type=int, default=128)
    ap.add_argument("--d_code", type=int, default=32)     # smaller code dim
    ap.add_argument("--n_codes", type=int, default=64)    # smaller codebook
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--entropy_w", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    train(args)
