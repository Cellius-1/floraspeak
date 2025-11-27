import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

def add_noise(sig, fs, hum=0.0, drift=0.0, gain_jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    T,C = sig.shape
    t = np.arange(T)/fs
    out = sig.copy()
    if hum>0:
        f = 60.0  # line hum
        out += hum*np.sin(2*np.pi*f*t)[:,None]
    if drift>0:
        k = drift * (rng.standard_normal((1,C)))
        out += (np.linspace(0,1,T)[:,None]) @ k
    if gain_jitter>0:
        g = 1.0 + rng.normal(0, gain_jitter, size=(1,C))
        out *= g
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_manifest", default="data/synth/manifest.csv")
    ap.add_argument("--out_root", default="data/synth_domains")
    args = ap.parse_args()

    man = pd.read_csv(args.base_manifest)

    # Domain A: copy as-is
    rootA = Path(args.out_root)/"A"; rootA.mkdir(parents=True, exist_ok=True)
    rowsA=[]
    for _,r in man.iterrows():
        sd = Path(r["path"]); od = rootA/sd.name; od.mkdir(exist_ok=True)
        for fname in ["meta.json","events.csv"]: shutil.copy(sd/fname, od/fname)
        shutil.copy(sd/"signals.csv", od/"signals.csv")
        rowsA.append({"uid": int(r["uid"]), "path": str(od)})
    pd.DataFrame(rowsA).to_csv(rootA/"manifest.csv", index=False)
    print(f"[OK] Domain A -> {rootA/'manifest.csv'}")

    # Domain B: add noise/drift/gain jitter
    rootB = Path(args.out_root)/"B"; rootB.mkdir(parents=True, exist_ok=True)
    rowsB=[]
    for i, r in man.iterrows():
        sd = Path(r["path"]); od = rootB/sd.name; od.mkdir(exist_ok=True)
        for fname in ["meta.json","events.csv"]: shutil.copy(sd/fname, od/fname)
        fs = json.load(open(sd/"meta.json"))["fs"]
        sig = pd.read_csv(sd/"signals.csv").values
        sig2 = add_noise(sig, fs, hum=0.02, drift=0.2, gain_jitter=0.05, seed=100+i)
        np.savetxt(od/"signals.csv", sig2, delimiter=",")
        rowsB.append({"uid": int(r["uid"]), "path": str(od)})
    pd.DataFrame(rowsB).to_csv(rootB/"manifest.csv", index=False)
    print(f"[OK] Domain B -> {rootB/'manifest.csv'}")
