import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    Cosine-distance EMA codebook.
    """
    def __init__(self, n_codes=64, d_code=32, beta=0.25, decay=0.97, eps=1e-5):
        super().__init__()
        self.n_codes, self.d_code = n_codes, d_code
        self.beta, self.decay, self.eps = beta, decay, eps
        self.codebook = nn.Parameter(torch.randn(n_codes, d_code))
        self.register_buffer("ema_cluster_size", torch.zeros(n_codes))
        self.register_buffer("ema_w", torch.randn(n_codes, d_code))

    def forward(self, z_e):
        # cosine VQ on L2-normalized vectors
        z = F.normalize(z_e, dim=1)
        cb = F.normalize(self.codebook, dim=1)
        with torch.no_grad():
            d = 2 - 2 * (z @ cb.t())   # [B,K]
            codes = torch.argmin(d, dim=1)
            onehot = F.one_hot(codes, self.n_codes).float()
            self.ema_cluster_size.mul_(self.decay).add_((1 - self.decay) * onehot.sum(0))
            dw = onehot.t() @ z
            self.ema_w.mul_(self.decay).add_((1 - self.decay) * dw)
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.n_codes * self.eps) * n)
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            embed_normalized = F.normalize(embed_normalized, dim=1)
            self.codebook.data.copy_(embed_normalized)
        z_q = F.normalize(self.codebook[codes], dim=1)
        vq_loss = self.beta * F.mse_loss(z_e.detach(), z_q)
        z_q = z_e + (z_q - z_e).detach()
        return z_q, codes, vq_loss

class VQVAE(nn.Module):
    """
    VQ-VAE on SSL embeddings with LayerNorm preconditioning.
    """
    def __init__(self, d_in=128, d_hidden=128, d_code=32, n_codes=64, beta=0.25):
        super().__init__()
        self.pre = nn.LayerNorm(d_in)  # stabilizes scale
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_code),
        )
        self.vq = VectorQuantizerEMA(n_codes=n_codes, d_code=d_code, beta=beta)
        self.dec = nn.Sequential(
            nn.Linear(d_code, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_in),
        )
    def forward(self, x):
        x = self.pre(x)
        z_e = F.normalize(self.enc(x), dim=1)
        z_q, codes, vq_loss = self.vq(z_e)
        x_hat = self.dec(z_q)
        rec = F.mse_loss(x_hat, x)
        loss = rec + vq_loss
        return x_hat, codes, loss, {"rec": rec.detach(), "vq": vq_loss.detach()}
