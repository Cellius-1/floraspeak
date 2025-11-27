import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

def scale_drought_signals(session_dir, out_dir, severity):
    sd = Path(session_dir); od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    # copy meta & events
    shutil.copy(sd/"meta.json", od/"meta.json")
    ev = pd.read_csv(sd/"events.csv")
    ev.to_csv(od/"events.csv", index=False)
    sig = pd.read_csv(sd/"signals.csv").values
    fs = json.load(open(sd/"meta.json"))["fs"]
    t = np.arange(sig.shape[0]) / fs
    # scale only drought intervals
    for _,e in ev.iterrows():
        if str(e["type"])!="drought": continue
        a = int(float(e["onset_s"])*fs)
        b = int((float(e["onset_s"])+float(e["duration_s"]))*fs)
        sig[a:b,:] = sig[a:b,:] * severity
    np.savetxt(od/"signals.csv", sig, delimiter=",")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_manifest", default="data/synth/manifest.csv")
    ap.add_argument("--out_root", default="data/synth_dose")
    ap.add_argument("--severities", type=float, nargs="+", default=[0.75,1.0,1.25,1.5])
    args = ap.parse_args()

    man = pd.read_csv(args.base_manifest)
    for sev in args.severities:
        out_manifest = []
        root = Path(args.out_root)/f"sev_{sev:.2f}"
        root.mkdir(parents=True, exist_ok=True)
        for _,row in man.iterrows():
            sd = Path(row["path"])
            od = root/sd.name
            od.mkdir(exist_ok=True)
            ok = scale_drought_signals(sd, od, sev)
            out_manifest.append({"uid": int(row["uid"]), "path": str(od)})
        pd.DataFrame(out_manifest).to_csv(root/"manifest.csv", index=False)
        print(f"[OK] wrote {root/'manifest.csv'}")
