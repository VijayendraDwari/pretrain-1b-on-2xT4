#!/usr/bin/env python3
"""Plot the validation NLL rows recorded in validation_curve.csv.

Requires matplotlib. Run from any directory with:
    python results/plot_validation.py
"""
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

updates = [int(row["update"]) for row in rows]
tokens_m = [int(row["tokens"]) / 1_000_000 for row in rows]
nll = [float(row["nll"]) for row in rows]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 11,
})
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#ffffff")
ax.plot(tokens_m, nll, color="#2867a8", linewidth=2.4, marker="o", markersize=6)
ax.set_title("Validation NLL during pretraining", loc="left", pad=14, weight="bold")
ax.set_xlabel("Model-input tokens processed (millions)")
ax.set_ylabel("Mean next-token NLL")
ax.set_xlim(left=0, right=max(tokens_m) * 1.08)
ax.set_ylim(bottom=0)
ax.grid(True, axis="both", color="#d9e1ea", linewidth=0.8, alpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#9aa8b7")
ax.spines["bottom"].set_color("#9aa8b7")

for i, (x, y, update) in enumerate(zip(tokens_m, nll, updates)):
    if update >= 616:
        dx = -42 if update == updates[-1] else 8
        dy = 10 if i % 2 == 0 else -18
        ax.annotate(f"update {update}", (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color="#34475a")

fig.text(0.01, 0.015,
         "Each point uses the trainer's fixed 131,072-token validation sample. "
         "This is a pretraining metric, not a downstream benchmark.",
         fontsize=8.5, color="#526273")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Wrote {OUTPUT_PNG}")
print(f"Wrote {OUTPUT_SVG}")
