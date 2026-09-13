"""
Draw the README banner.

The artwork is not decoration - it is the project's own output. The trace
behind the wordmark is the real autoencoder reconstruction error over the
held-out drive, and the shaded bands are the real attack episodes.

    python tools/make_banner.py
"""

import json
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from canguard import models

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, MODELS, DOCS = ROOT / "data", ROOT / "models", ROOT / "docs"

BG = "#12161a"
INK = "#f3f5f2"
DIM = "#79838b"
CAN_H = "#e3b81f"        # CAN-High is yellow in most harnesses
CAN_L = "#3ca773"        # CAN-Low is green
BAND = "#2a3138"


def spaced(ax, x, y, text, size, color, weight, tracking, **kw):
    """Draw letter-spaced text - matplotlib has no tracking control."""
    t = ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
                family="DejaVu Sans", transform=ax.transAxes, **kw)
    try:
        import matplotlib.textpath  # noqa: F401
        t.set_fontstretch("condensed")
    except Exception:
        pass
    return t


def main():
    DOCS.mkdir(exist_ok=True)

    z = np.load(DATA / "test_features.npz", allow_pickle=True)
    X, starts = z["X"], z["starts"]
    bundle = joblib.load(MODELS / "scaler.joblib")
    meta = json.load(open(MODELS / "meta.json"))

    from tensorflow import keras
    ae = keras.models.load_model(MODELS / "autoencoder.keras")
    err = models.recon_error(ae, models.scale(bundle["scaler"], X))

    eps = []
    p = DATA / "test_episodes.csv"
    if p.exists():
        e = pd.read_csv(p)
        eps = list(zip(e.start_s, e.end_s))

    fig = plt.figure(figsize=(12.8, 3.4), dpi=200)
    fig.patch.set_facecolor(BG)

    # --- the trace, full-bleed across the right two thirds ------------------
    ax = fig.add_axes([0.40, 0.16, 0.58, 0.70])
    ax.set_facecolor(BG)
    for s, e2 in eps:
        ax.axvspan(s, e2, color=BAND, zorder=1)
    ax.plot(starts, np.log10(np.maximum(err, 1e-3)), color=CAN_H, lw=0.9, zorder=3)
    ax.axhline(np.log10(meta["ae_threshold"]), color=CAN_L, lw=1.1,
               ls=(0, (4, 3)), zorder=4)
    ax.set_xlim(starts.min(), starts.max())
    ax.set_ylim(-1.2, 2.7)
    ax.axis("off")

    # fade the trace in from the left so it reads as one field, not a pasted panel
    pos = ax.get_position()
    fade = fig.add_axes([pos.x0, pos.y0, pos.width * 0.28, pos.height], zorder=6)
    fade.imshow(np.linspace(1.0, 0.0, 512)[None, :], aspect="auto",
                extent=[0, 1, 0, 1], vmin=0, vmax=1,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                    "fade", [(0, 0, 0, 0), BG]))
    fade.axis("off")
    fade.patch.set_alpha(0)

    # --- wordmark ----------------------------------------------------------
    tx = fig.add_axes([0, 0, 1, 1])
    tx.axis("off")
    tx.set_facecolor("none")

    tx.text(0.045, 0.615, "CAN-GUARD", fontsize=44, color=INK, fontweight="bold",
            family="DejaVu Sans", va="center", ha="left")
    tx.text(0.047, 0.375,
            "MACHINE-LEARNING INTRUSION DETECTION\nFOR AUTOMOTIVE CAN-BUS NETWORKS",
            fontsize=9.2, color=DIM, family="DejaVu Sans Mono",
            va="center", ha="left", linespacing=1.85)

    # a twisted-pair rule: CAN-H over CAN-L
    tx.add_patch(Rectangle((0.047, 0.205), 0.145, 0.016,
                           facecolor=CAN_H, transform=tx.transAxes, clip_on=False))
    tx.add_patch(Rectangle((0.047, 0.183), 0.145, 0.016,
                           facecolor=CAN_L, transform=tx.transAxes, clip_on=False))

    stats = json.load(open(ROOT / "reports" / "metrics.json"))
    best = max(stats["overall"].values(), key=lambda v: v["f1"])
    line = (f"{best['f1']:.3f} F1   ·   {100 * best['false_alarm_rate']:.2f}% "
            f"FALSE ALARMS   ·   {len(stats['attack_families'])} ATTACK FAMILIES")
    tx.text(0.047, 0.085, line,
            fontsize=7.6, color=CAN_H, family="DejaVu Sans Mono",
            va="center", ha="left", alpha=.9)

    out = DOCS / "banner.png"
    fig.savefig(out, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
