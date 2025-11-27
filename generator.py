#!/usr/bin/env python3
"""
Synthetic Plant Electrophysiology Generator (FloraVoice)

Creates realistic multichannel time series with:
 - Multiple species and plants
 - Randomized stimulus schedules (light, touch, heat, cold, nutrient, drought)
 - Ground-truth latent states (hydration, excitability, temperature, nutrient level)
 - Channel topology (leaf/stem/root) with distinct response kernels
 - Realistic artefacts: line hum, drift, pink+white noise, dropouts, saturation
 - Circadian modulation
 - Rich metadata & manifest for ML experiments

Dependencies: numpy, pandas (pip install numpy pandas)
"""

from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- Utilities ------------------------------------- #

def set_seed(seed: int | None):
    if seed is None: return
    np.random.seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rand_between(a, b):
    return a + (b - a) * np.random.rand()

# --------------------------- Latent Dynamics --------------------------------- #

def simulate_latents(T, fs, circadian_phase=0.0):
    """
    Simulate hidden plant state variables over time:
      - hydration (0..1): slow drift; drought reduces it
      - excitability (0..1): modulates spike amplitude/shape
      - temperature (°C): baseline + slow variation
      - nutrient (0..1): step changes w/ nutrient events
      - circadian (sinusoid 24h mapped to recording length)
    Returns dict of arrays with length T.
    """
    t = np.arange(T) / fs

    # Circadian proxy over the recording window (rescaled 24h → duration)
    circadian = 0.5 + 0.5 * np.sin(2 * np.pi * (t / t[-1] + circadian_phase))

    # Baselines with colored noise
    def smooth_noise(scale=1.0, tau=30.0):
        x = np.random.randn(T) * scale
        y = np.zeros(T)
        a = np.exp(-1.0 / (fs * tau))
        for i in range(1, T):
            y[i] = a * y[i-1] + (1 - a) * x[i]
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        return y

    hydration   = 0.7 * np.ones(T) + 0.2 * smooth_noise(0.5, tau=120)
    excitability= 0.6 * np.ones(T) + 0.25* smooth_noise(0.8, tau=20)
    temperature = 24 + 1.5 * np.sin(2*np.pi*t / (t[-1] + 1e-8)) + 0.5*np.random.randn(T)
    nutrient    = 0.6 * np.ones(T) + 0.15* smooth_noise(0.6, tau=60)

    latents = {
        "time": t,
        "hydration": np.clip(hydration, 0, 1),
        "excitability": np.clip(excitability, 0, 1),
        "temperature": temperature,
        "nutrient": np.clip(nutrient, 0, 1),
        "circadian": np.clip(circadian, 0, 1),
    }
    return latents

# ---------------------------- Stimulus Model --------------------------------- #

STIMULI = ["light", "touch", "heat", "cold", "nutrient", "drought"]

def random_schedule(duration, fs, min_isi=15.0, max_isi=60.0,
                    max_events=8, allowed=STIMULI):
    """
    Create a randomized stimulus schedule (list of dicts).
    Each event has onset (sec), type, intensity in [0,1], and duration (sec).
    """
    events = []
    t = rand_between(5.0, 10.0)  # first event after small delay
    while t < duration - 5 and len(events) < max_events:
        stim = np.random.choice(allowed)
        intensity = rand_between(0.3, 1.0)
        # Typical durations
        dur = {
            "light": rand_between(1.0, 5.0),
            "touch": rand_between(0.1, 0.5),
            "heat":  rand_between(5.0, 20.0),
            "cold":  rand_between(5.0, 20.0),
            "nutrient": rand_between(3.0, 8.0),
            "drought":  rand_between(15.0, 60.0),
        }[stim]
        events.append({
            "onset_s": float(t),
            "type": stim,
            "intensity": float(intensity),
            "duration_s": float(dur)
        })
        t += rand_between(min_isi, max_isi)
    return events

# ----------------------- Channel Topology & Kernels -------------------------- #

def channel_layout(n_channels):
    """
    Assign channel roles to mimic electrode placement:
      - leaf, stem, root (cyclic assignment)
    """
    roles = []
    for i in range(n_channels):
        roles.append(["leaf", "stem", "root"][i % 3])
    return roles

def response_kernel(stim_type, role, fs, intensity, excitability):
    """
    Generate a role-specific impulse response for a given stimulus.
    Returns a short kernel to be convolved/added around event onset.
    """
    # Base lengths
    base_len_s = {
        "light": 2.0,
        "touch": 0.8,
        "heat":  6.0,
        "cold":  6.0,
        "nutrient": 3.5,
        "drought": 10.0
    }[stim_type]
    L = max(int(base_len_s * fs), 10)
    t = np.linspace(0, base_len_s, L)

    # Role scaling
    role_gain = {"leaf": 1.0, "stem": 0.7, "root": 0.8}[role]

    # Shape per stimulus
    if stim_type == "light":
        # fast rise then decay (leaf-dominant)
        kernel = np.exp(-3*t) * (1 + 2*np.exp(-((t-0.1)/0.05)**2))
    elif stim_type == "touch":
        # biphasic sharp transient
        kernel = np.exp(-8*t) - 0.6*np.exp(-18*t)
    elif stim_type == "heat":
        kernel = 0.5*np.tanh(3*t) * np.exp(-0.3*t)
    elif stim_type == "cold":
        kernel = -0.5*np.tanh(3*t) * np.exp(-0.3*t)
    elif stim_type == "nutrient":
        kernel = np.exp(-1.5*t) * (1 + 0.7*np.sin(2*np.pi*2.0*t))
    elif stim_type == "drought":
        # slow, prolonged shift
        kernel = (t / (base_len_s + 1e-6)) * np.exp(-0.15*t)
    else:
        kernel = np.zeros_like(t)

    # Amplitude scales with intensity * excitability * role_gain
    amp = (0.4 + 0.6*intensity) * (0.4 + 0.6*excitability) * role_gain
    return amp * kernel

# ----------------------- Artefacts & Background ------------------------------ #

def line_hum(T, fs, mains=60.0, harmonics=3, amp=0.003):
    t = np.arange(T) / fs
    y = np.zeros(T)
    for k in range(1, harmonics+1):
        y += (amp / k) * np.sin(2*np.pi*mains*k*t + 2*np.pi*np.random.rand())
    return y

def pink_noise(T, scale=0.01):
    # Voss-McCartney-like: average of random walks
    n_layers = 8
    y = np.zeros(T)
    for _ in range(n_layers):
        step = np.random.randn(T).cumsum()
        y += step
    y = (y - y.mean()) / (y.std() + 1e-8)
    return scale * y

def slow_drift(T, fs, amp=0.02, tau=300.0):
    y = np.zeros(T)
    a = np.exp(-1.0 / (fs * tau))
    for i in range(1, T):
        y[i] = a*y[i-1] + (1-a)*np.random.randn()*amp
    return y

def induce_dropouts(sig, fs, p=0.002, max_len_s=1.0):
    T = len(sig)
    i = 0
    while i < T:
        if np.random.rand() < p:
            L = int(rand_between(0.05, max_len_s) * fs)
            sig[i:i+L] = 0.0
            i += L
        else:
            i += 1
    return sig

def induce_saturation(sig, thresh=3.0):
    sig = sig.copy()
    sig[sig >  thresh] =  thresh
    sig[sig < -thresh] = -thresh
    return sig

# ------------------------------ Generator ------------------------------------ #

def generate_recording(fs=1000, duration=600, n_channels=8, species="A",
                       plant_id=0, session_id=0, circadian_phase=None,
                       schedule=None, seed=None):
    """
    Generate one recording with signals, events, and latents.
    Returns dict with signals_df, events_df, latents_df, meta
    """
    set_seed(seed)
    T = int(duration * fs)
    roles = channel_layout(n_channels)

    if circadian_phase is None:
        circadian_phase = np.random.rand()

    # Latent states
    lat = simulate_latents(T, fs, circadian_phase=circadian_phase)

    # Stimulus schedule
    if schedule is None:
        schedule = random_schedule(duration, fs)

    # Base signals per channel: slow oscillations + noise
    t = lat["time"]
    signals = np.zeros((T, n_channels))
    base_amp = 0.08 + 0.04*np.random.rand(n_channels)
    slow_freq = 0.02 + 0.05*np.random.rand(n_channels)

    for ch in range(n_channels):
        # baseline oscillation (circadian modulated)
        baseline = base_amp[ch] * (0.5 + lat["circadian"]) * np.sin(2*np.pi*slow_freq[ch]*t + 2*np.pi*np.random.rand())
        # complexity bursts tied to excitability
        bursts = 0.03 * (0.5 + 0.8*lat["excitability"]) * np.sin(2*np.pi*(2.0+0.5*np.random.rand())*t + 2*np.pi*np.random.rand())
        signals[:, ch] = baseline + bursts

    # Apply stimuli responses
    for ev in schedule:
        onset_idx = int(ev["onset_s"] * fs)
        dur_len   = int(ev["duration_s"] * fs)
        idx_end   = min(T, onset_idx + dur_len + int(0.5*fs))
        if onset_idx >= T: continue

        for ch in range(n_channels):
            role = roles[ch]
            # excitability snapshot ~ local mean around onset
            i0 = max(0, onset_idx - int(1.0*fs))
            i1 = min(T-1, onset_idx + int(1.0*fs))
            exc_local = float(np.clip(lat["excitability"][i0:i1].mean(), 0, 1))
            kernel = response_kernel(ev["type"], role, fs, ev["intensity"], exc_local)
            L = len(kernel)

            j0 = onset_idx
            j1 = min(T, onset_idx + L)
            signals[j0:j1, ch] += kernel[:(j1 - j0)]

            # Event-specific latent nudges
            if ev["type"] == "drought":
                lat["hydration"][onset_idx:idx_end] -= 0.10 * ev["intensity"]
            if ev["type"] == "nutrient":
                lat["nutrient"][onset_idx:idx_end] += 0.08 * ev["intensity"]
            if ev["type"] == "heat":
                lat["temperature"][onset_idx:idx_end] += 2.0 * ev["intensity"]
            if ev["type"] == "cold":
                lat["temperature"][onset_idx:idx_end] -= 2.0 * ev["intensity"]

    # Add artefacts & measurement effects
    for ch in range(n_channels):
        signals[:, ch] += line_hum(T, fs, mains=60.0, harmonics=3, amp=0.002 + 0.001*np.random.rand())
        signals[:, ch] += pink_noise(T, scale=0.01 + 0.01*np.random.rand())
        signals[:, ch] += slow_drift(T, fs, amp=0.015 + 0.01*np.random.rand(), tau=200 + 150*np.random.rand())
        # occasional dropouts & saturation
        if np.random.rand() < 0.3:
            signals[:, ch] = induce_dropouts(signals[:, ch], fs, p=0.0015, max_len_s=0.7)
        signals[:, ch] = induce_saturation(signals[:, ch], thresh=3.5)

    # Standardize per channel (robust-ish)
    for ch in range(n_channels):
        med = np.median(signals[:, ch])
        mad = np.median(np.abs(signals[:, ch] - med)) + 1e-6
        signals[:, ch] = (signals[:, ch] - med) / (1.4826 * mad)

    # Assemble DataFrames
    signals_df = pd.DataFrame(signals, columns=[f"ch{c}" for c in range(n_channels)])
    signals_df.insert(0, "time", t)

    events_df = pd.DataFrame(schedule)
    if not events_df.empty:
        events_df["session_time_s"] = events_df["onset_s"]  # duplicate column for convenience

    latents_df = pd.DataFrame({k:v for k,v in lat.items()})

    meta = {
        "fs": fs,
        "duration_s": duration,
        "n_channels": n_channels,
        "species": species,
        "plant_id": int(plant_id),
        "session_id": int(session_id),
        "roles": channel_layout(n_channels),
        "circadian_phase": float(circadian_phase),
        "stimuli": STIMULI
    }
    return {
        "signals_df": signals_df,
        "events_df": events_df,
        "latents_df": latents_df,
        "meta": meta
    }

# ------------------------------ Batch Runner --------------------------------- #

def run_batch(outdir="data/synth", species=("A","B"),
              plants_per_species=2, sessions_per_plant=2,
              fs=1000, duration=600, channels=8, seed=123):
    set_seed(seed)
    outdir = Path(outdir)
    ensure_dir(outdir)

    manifest_rows = []
    uid = 0

    for sp in species:
        for p in range(plants_per_species):
            for s in range(sessions_per_plant):
                circadian_phase = np.random.rand()
                rec = generate_recording(
                    fs=fs, duration=duration, n_channels=channels,
                    species=sp, plant_id=p, session_id=s,
                    circadian_phase=circadian_phase, seed=np.random.randint(0, 1_000_000)
                )

                # Write to disk
                rec_dir = outdir / f"species_{sp}" / f"plant_{p:03d}" / f"session_{s:03d}"
                ensure_dir(rec_dir)

                rec["signals_df"].to_csv(rec_dir/"signals.csv", index=False)
                rec["latents_df"].to_csv(rec_dir/"latents.csv", index=False)
                if len(rec["events_df"]) > 0:
                    rec["events_df"].to_csv(rec_dir/"events.csv", index=False)
                else:
                    pd.DataFrame(columns=["onset_s","type","intensity","duration_s","session_time_s"]).to_csv(rec_dir/"events.csv", index=False)

                with open(rec_dir/"meta.json", "w") as f:
                    json.dump(rec["meta"], f, indent=2)

                manifest_rows.append({
                    "uid": uid,
                    "species": sp,
                    "plant_id": p,
                    "session_id": s,
                    "path": str(rec_dir),
                    "fs": rec["meta"]["fs"],
                    "duration_s": rec["meta"]["duration_s"],
                    "n_channels": rec["meta"]["n_channels"],
                    "circadian_phase": rec["meta"]["circadian_phase"]
                })
                uid += 1

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(outdir/"manifest.csv", index=False)
    return outdir, manifest

# ------------------------------- CLI ----------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="FloraVoice Synthetic Plant Generator")
    ap.add_argument("--outdir", type=str, default="data/synth", help="Output directory")
    ap.add_argument("--species", type=str, default="A,B", help="Comma-separated species labels")
    ap.add_argument("--plants-per-species", type=int, default=2)
    ap.add_argument("--sessions-per-plant", type=int, default=2)
    ap.add_argument("--fs", type=int, default=1000, help="Sampling rate (Hz)")
    ap.add_argument("--duration", type=float, default=600, help="Duration per session (sec)")
    ap.add_argument("--channels", type=int, default=8, help="Number of channels")
    ap.add_argument("--seed", type=int, default=123, help="Global seed")
    return ap.parse_args()

def main():
    args = parse_args()
    sp_list = [s.strip() for s in args.species.split(",") if s.strip()]
    outdir, manifest = run_batch(
        outdir=args.outdir, species=tuple(sp_list),
        plants_per_species=args.plants_per_species,
        sessions_per_plant=args.sessions_per_plant,
        fs=args.fs, duration=args.duration,
        channels=args.channels, seed=args.seed
    )
    print(f"[OK] Wrote synthetic dataset to: {outdir}")
    print(f"     Sessions: {len(manifest)}")
    print(f"     Example row:\n{manifest.head(1).to_string(index=False)}")

if __name__ == "__main__":
    main()
