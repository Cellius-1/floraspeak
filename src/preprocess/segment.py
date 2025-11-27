import numpy as np

def sliding_windows(x, win_s, hop_s, fs):
    W = int(win_s * fs)
    H = int(hop_s * fs)
    T, C = x.shape
    idx = [(i, i + W) for i in range(0, T - W + 1, H)]
    arr = np.stack([x[i:j] for i, j in idx], axis=0)  # [N, W, C]
    return arr, idx
