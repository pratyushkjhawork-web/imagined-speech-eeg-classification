"""Aggregate results/ into a comparison table + per-subject plot + confusion matrices.
Usage: python scripts/report.py --results results
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(f"{a.results}/*_summary.json")):
        j = json.load(open(f)); ms = j["mean_std"]
        rows.append({"dataset": j.get("dataset", "?"), "model": j["model"], "seed": j["args"]["seed"], "params": j["params"],
                     **{k: f"{v[0]*100:.2f} ± {v[1]*100:.2f}" for k, v in ms.items()}})
    t = pd.DataFrame(rows); print(t.to_string(index=False))
    t.to_csv(f"{a.results}/comparison_table.csv", index=False)

    plt.figure(figsize=(8, 4))
    for f in sorted(glob.glob(f"{a.results}/*_per_subject.csv")):
        d = pd.read_csv(f)
        plt.plot(d["test_subject"], d["accuracy"], marker="o",
                 label=os.path.basename(f).replace("_per_subject.csv", ""))
    plt.xlabel("held-out subject"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{a.results}/per_subject_accuracy.png", dpi=200)

    for f in sorted(glob.glob(f"{a.results}/*_confusion.npy")):
        cm = np.load(f).astype(float); cm /= cm.sum(1, keepdims=True).clip(min=1)
        plt.figure(figsize=(5, 4.5)); plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(); plt.xlabel("predicted"); plt.ylabel("true")
        plt.title(os.path.basename(f).replace("_confusion.npy", "")); plt.tight_layout()
        plt.savefig(f.replace(".npy", ".png"), dpi=200)


if __name__ == "__main__":
    main()
