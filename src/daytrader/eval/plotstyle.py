"""Shared matplotlib style — dataviz reference palette, light mode.

Categorical slots are assigned in FIXED order (never cycled ad hoc):
slot 1 blue = real training, slot 2 aqua = real validation,
synthetic universes are drawn as a muted family (identity is real-vs-synth).
Status colors are reserved for pass/fail semantics only.
"""
import matplotlib as mpl

SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"   # status — reserved
SYNTH_FAMILY = "#b5b3ab"   # muted family color for synthetic universes


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 150,
            "figure.autolayout": True,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "font.family": "sans-serif",
        }
    )
