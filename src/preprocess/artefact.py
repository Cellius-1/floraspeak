import numpy as np

def zscore_clip(x, z=6.0):
    m = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - m), axis=0, keepdims=True) + 1e-9
    zed = (x - m) / (1.4826 * mad)
    mask = (np.abs(zed) > z).any(axis=1)  # timepoints (rows) with any channel outlier
    x = x.copy()
    x[mask] = 0.0
    return x, mask

def bad_window_mask(windows, frac_bad=0.2):
    zero_frac = (windows == 0).mean(axis=(1, 2))
    return zero_frac > frac_bad
