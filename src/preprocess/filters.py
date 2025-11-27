import numpy as np
from scipy.signal import iirnotch, filtfilt, butter

def notch_line(x, fs, f0=60.0, Q=30.0, harmonics=3):
    """Apply notch filters at f0 and its first `harmonics` multiples. x: [T, C]."""
    x = np.asarray(x, float)
    y = x.copy()
    ny = fs / 2.0
    for k in range(1, harmonics + 1):
        w0 = (f0 * k) / ny
        if not (0 < w0 < 1):  # guard against Nyquist issues
            continue
        b, a = iirnotch(w0=w0, Q=Q)
        y = filtfilt(b, a, y, axis=0, padlen=150)
    return y

def bandpass(x, fs, lo=0.1, hi=120.0, order=4):
    """Butterworth bandpass filter."""
    x = np.asarray(x, float)
    ny = fs / 2.0
    lo_ny = max(lo, 1e-6) / ny
    hi_ny = min(hi, ny - 1e-3) / ny
    if hi_ny <= lo_ny:
        return x
    b, a = butter(order, [lo_ny, hi_ny], btype="bandpass")
    return filtfilt(b, a, x, axis=0, padlen=150)

def robust_detrend(x, win_secs=30.0, fs=1000, method="ema"):
    """
    Low-memory detrend.
    method='ema'  : subtract first-order IIR low-pass (time-constant ~= win_secs/2)
    method='median': safe only for small windows (<= 5001 samples) to avoid huge memory.
    """
    x = np.asarray(x, float)
    T, C = x.shape

    if method == "median":
        k = int(win_secs * fs)
        if k % 2 == 0:
            k += 1
        if k > 5001:
            # fall back to EMA for large kernels to avoid huge memory
            method = "ema"
        else:
            from scipy.signal import medfilt
            y = np.empty_like(x)
            for c in range(C):
                base = medfilt(x[:, c], kernel_size=k)
                y[:, c] = x[:, c] - base
            return y

    # EMA baseline (first-order low-pass), tiny memory, fast
    tau = max(1.0, win_secs / 2.0)            # time-constant in seconds
    alpha = np.exp(-1.0 / (fs * tau))         # 0<alpha<1
    y = np.empty_like(x)
    for c in range(C):
        v = x[:, c]
        base = np.empty_like(v)
        base[0] = v[0]
        for i in range(1, T):
            base[i] = alpha * base[i-1] + (1 - alpha) * v[i]
        y[:, c] = v - base
    return y
