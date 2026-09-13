"""Chart styling and figures for the evaluation report."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec",
             "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("cg_blue", BLUE_RAMP)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 30,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
    })


def _clean(ax, xgrid=False, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)


def _sub(ax, text):
    """Subtitle sits between the title and the plot frame."""
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK2,
            fontsize=8.5, ha="left", va="bottom")


def _legend_below(ax, ncol=2):
    """Legend under the axis, for charts whose title line is already busy."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=ncol,
              handlelength=1.4, columnspacing=1.6, fontsize=8)


def _legend_above(ax, ncol=3):
    """Legend on the title line, right-aligned - never over the data."""
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=ncol,
              handlelength=1.6, columnspacing=1.4)


# --------------------------------------------------------------------------

def roc_curves(curves, path, title, subtitle):
    """curves: list of (label, fpr, tpr, auc)"""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls=(0, (4, 4)), zorder=1)
    for i, (lab, fpr, tpr, auc) in enumerate(curves):
        ax.plot(fpr, tpr, color=SERIES[i], label=f"{lab}  (AUC {auc:.3f})", zorder=3)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    _sub(ax, subtitle)
    ax.legend(loc="lower right")
    _clean(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confusion(cm, labels, path, title, subtitle):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}\n{100*norm[i, j]:.1f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _sub(ax, subtitle)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(categories, series, path, title, subtitle, ylabel):
    """series: list of (label, values 0..1)"""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    n = len(series)
    x = np.arange(len(categories))
    w = 0.80 / n
    for i, (lab, vals) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(x + off, vals, width=w * 0.92, color=SERIES[i],
                      label=lab, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{100*v:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _sub(ax, subtitle)
    _legend_below(ax, ncol=4)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def importance(names, vals, path, title, subtitle):
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=SERIES[0], height=0.68, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:.3f}", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(title)
    _sub(ax, subtitle)
    _clean(ax, xgrid=True, ygrid=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def anomaly_timeline(t, err, thr, episodes, path, title, subtitle):
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    seen = set()
    for _name, s, e in episodes:
        ax.axvspan(s, e, color="#e1e0d9", alpha=0.85, zorder=1,
                   label="attack window" if "w" not in seen else None)
        seen.add("w")
    ax.plot(t, err, color=SERIES[0], lw=1.2, zorder=3, label="reconstruction error")
    ax.axhline(thr, color=CRITICAL, lw=1.6, ls=(0, (5, 3)), zorder=4,
               label=f"alarm threshold ({thr:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("time in trace (s)")
    ax.set_ylabel("mean squared reconstruction error")
    ax.set_title(title)
    _sub(ax, subtitle)
    _legend_above(ax, ncol=3)
    ax.set_xlim(t.min(), t.max())
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
