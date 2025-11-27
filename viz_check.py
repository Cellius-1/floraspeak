import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

root = Path("data/synth")
manifest = pd.read_csv(root/"manifest.csv")
row = manifest.iloc[0]
rec_dir = Path(row["path"])

signals = pd.read_csv(rec_dir/"signals.csv")
events  = pd.read_csv(rec_dir/"events.csv")
with open(rec_dir/"meta.json") as f: meta = json.load(f)

# plot one channel + event onsets
ch = "ch0"
plt.figure(figsize=(12,4))
plt.plot(signals["time"], signals[ch], linewidth=0.8)
for _,e in events.iterrows():
    x = e["onset_s"]
    plt.axvline(x, linestyle="--", alpha=0.5)
    plt.text(x, signals[ch].max()*0.8, e["type"], rotation=90, fontsize=8)
plt.title(f"{rec_dir.name} • {ch} • fs={meta['fs']}Hz")
plt.xlabel("Time (s)"); plt.ylabel("z-scored signal")
plt.tight_layout(); plt.show()
