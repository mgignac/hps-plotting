"""HPS matplotlib style configuration."""

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_hps_style():
    """Apply HPS publication-quality matplotlib style."""
    params = {
        # Figure
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "figure.facecolor": "white",

        # Font
        "font.family": "sans-serif",
        "font.size": 14,
        "mathtext.fontset": "dejavusans",

        # Axes
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "axes.linewidth": 1.2,
        "axes.formatter.use_mathtext": True,

        # Ticks
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # Legend
        "legend.fontsize": 12,
        "legend.frameon": False,
        "legend.handlelength": 1.5,

        # Lines
        "lines.linewidth": 1.5,

        # Histogram
        "hist.bins": 50,

        # Errorbar
        "errorbar.capsize": 0,

        # Savefig
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    }
    mpl.rcParams.update(params)


def add_hps_label(ax, label="HPS", sublabel="Internal", x=0.05, y=0.95,
                  lumi=None, extra_lines=None):
    """Add the HPS experiment label to a plot axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label : str
        Main label (bold).
    sublabel : str
        Sublabel (italic).
    x, y : float
        Position in axes coordinates.
    lumi : float, optional
        Integrated luminosity in pb^{-1}.  When provided, a line
        showing ``L_int = XX pb^{-1}`` (2 significant figures) is drawn
        below the main label.
    extra_lines : list of str, optional
        Additional text lines rendered below the lumi line (or the main
        label when lumi is None), e.g. a selection or hit-category label.
    """
    ax.text(
        x, y,
        f"$\\bf{{{label}}}$ {sublabel}",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    y_cursor = y - 0.07
    if lumi is not None:
        ax.text(
            x, y_cursor,
            f"$L_\\mathrm{{int}} = {lumi:.2g}\\,\\mathrm{{pb}}^{{-1}}$",
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="top",
        )
        y_cursor -= 0.07
    for line in (extra_lines or []):
        ax.text(
            x, y_cursor,
            line,
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="top",
        )
        y_cursor -= 0.07
