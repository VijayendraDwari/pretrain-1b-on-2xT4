#!/usr/bin/env python3
"""Plot canonical validation NLL rows from validation_curve.csv."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "validation_curve.csv"
OUTPUT_PNG = ROOT / "validation_nll.png"
OUTPUT_SVG = ROOT / "validation_nll.svg"

with DATA.open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"No validation rows found in {DATA}")

updates = [int(r["update"]) for r in rows]
tokens_m = [int(r["tokens"]) / 1_000_000 for r in rows]
nll = [float(r["nll"]) for r in rows]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 11,
})
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#ffffff")
ax.plot(tokens_m, nll, color="#2867a8", linewidth=2.4, marker="o", markersize=4.8)
ax.set_title("Validation NLL during pretraining", loc="left", pad=14, weight="bold")
ax.set_xlabel("Model-input tokens processed (millions)")
ax.set_ylabel("Mean next-token NLL")
ax.set_xlim(left=0, right=max(tokens_m) * 1.06)
ax.set_ylim(bottom=0)
ax.grid(True, axis="both", color="#d9e1ea", linewidth=0.8, alpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#9aa8b7")
ax.spines["bottom"].set_color("#9aa8b7")

label_updates = {616, 1225, 1829, 2436, 3031, 3600}
for i, (x, y, update) in enumerate(zip(tokens_m, nll, updates)):
    if update in label_updates:
        dx = -48 if update == updates[-1] else 7
        dy = 10 if i % 2 == 0 else -18
        ax.annotate(f"u{update}", (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5, color="#34475a")

fig.text(
    0.01, 0.015,
    "Each point uses the trainer's fixed 131,072-token validation sample. "
    "Latest measured point: u3600 / 471.86M tokens / NLL 3.69297.",
    fontsize=8.5, color="#526273",
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Wrote {OUTPUT_PNG}")
print(f"Wrote {OUTPUT_SVG}")
