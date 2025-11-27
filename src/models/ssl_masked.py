import math, torch, torch.nn as nn, torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d]
    def forward(self, x):
        # x: [B,L,d]
        return x + self.pe[:, :x.size(1), :]

class MaskedAutoencoderTS(nn.Module):
    """
    Simple masked autoencoder for multichannel time series.
    - Conv1D stem (per channel -> d_model)
    - Transformer encoder
    - Linear head to reconstruct masked time points
    Also exposes an embedding via mean-pooled encoder outputs.
    """
    def __init__(self, n_channels, d_model=128, n_heads=4, n_layers=4, p_drop=0.1):
        super().__init__()
        self.stem = nn.Conv1d(n_channels, d_model, kernel_size=7, padding=3)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                   dim_feedforward=4*d_model,
                                                   dropout=p_drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.posenc = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, n_channels)  # predict all channels at masked steps

    def forward(self, x, mask_idx=None):
        """
        x: [B, L, C]
        mask_idx: list of tensors length B with positions to reconstruct
        Returns:
          pred_masked: reconstructed values at masked idx, shape [total_masked, C]
          target_masked: ground truth at masked idx, shape [total_masked, C]
          z: encoder sequence features [B,L,d_model]
        """
        B, L, C = x.shape
        h = self.stem(x.transpose(1,2)).transpose(1,2)  # [B,L,d]
        h = self.posenc(h)
        z = self.encoder(h)                              # [B,L,d]
        y = self.head(z)                                 # [B,L,C]

        if mask_idx is None:
            return y, None, z

        preds, targs = [], []
        for b in range(B):
            idx = mask_idx[b]
            preds.append(y[b, idx])       # [M_b, C]
            targs.append(x[b, idx])       # [M_b, C]
        pred_masked = torch.cat(preds, 0)
        target_masked = torch.cat(targs, 0)
        return pred_masked, target_masked, z

    @torch.no_grad()
    def embed(self, x):
        """ Return mean-pooled encoder embeddings [B, d_model] """
        h = self.stem(x.transpose(1,2)).transpose(1,2)
        h = self.posenc(h)
        z = self.encoder(h)
        return z.mean(dim=1)
