#!/usr/bin/env python3
"""Plot run-by-run z0 fit results from the JSON produced by the binned plot framework.

Two output files are produced — one per detector half (top / bottom).
Each canvas has two panels: slope (left) and intercept (right),
with electron and positron shown as separate series.

Usage:
    python plot_z0_fits.py plots/z0_tanl_perrun/z0_fits.json
    python plot_z0_fits.py plots/z0_tanl_perrun/z0_fits.json --output-dir my_dir/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "ele": dict(color="royalblue", marker="o", ls="-",  label="Electron"),
    "pos": dict(color="firebrick", marker="s", ls="-",  label="Positron"),
}

HALVES = {
    "top": r"$\tan\lambda > 0$ (top detector)",
    "bot": r"$\tan\lambda < 0$ (bottom detector)",
}


def load(path):
    with open(path) as f:
        data = json.load(f)
    runs = sorted(data.keys(), key=lambda r: int(r) if r.isdigit() else r)
    return runs, data


def _collect(data, runs, track, half, param):
    """Return (x_indices, values) for the given track/half/param combination."""
    key = f"{track}_{half}"
    xs, ys = [], []
    for i, run in enumerate(runs):
        entry = data[run].get(key, {})
        if param in entry:
            xs.append(i)
            ys.append(entry[param])
    return np.array(xs), np.array(ys)


def _dynamic_ylim(ax, all_vals, min_span=None):
    if not all_vals:
        return
    lo, hi = min(all_vals), max(all_vals)
    pad = 0.35 * (hi - lo) if hi != lo else 0.05
    y_lo = lo - pad
    y_hi = hi + pad
    if min_span is not None:
        centre = 0.5 * (y_lo + y_hi)
        half_span = max(0.5 * (y_hi - y_lo), min_span / 2)
        y_lo = centre - half_span
        y_hi = centre + half_span
    ax.set_ylim(y_lo, y_hi)


def plot_half(half, half_title, runs, data, output_path):
    params     = ["slope",         "intercept"]
    ylabels    = ["Slope $a$",     "Intercept $b$ [mm]"]
    min_spans  = [None,             0.04]   # intercept: minimum ±0.02

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    fig.subplots_adjust(wspace=0.32)
    fig.suptitle(f"$z_0$ fit parameters — {half_title}", fontsize=12)

    xs_ticks = np.arange(len(runs))

    for ax, param, ylabel, min_span in zip(axes, params, ylabels, min_spans):
        all_vals = []
        for track in ("ele", "pos"):
            xs, ys = _collect(data, runs, track, half, param)
            if len(xs):
                sty = STYLES[track]
                ax.plot(xs, ys,
                        color=sty["color"], marker=sty["marker"],
                        ls=sty["ls"], label=sty["label"],
                        markersize=4, lw=1.2)
                all_vals.extend(ys.tolist())

        ax.axhline(0.0, color="gray", lw=0.7, ls=":")
        _dynamic_ylim(ax, all_vals, min_span=min_span)
        ax.set_xticks(xs_ticks)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Run number", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9, loc="best")
        ax.tick_params(axis="both", labelsize=8)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="Path to z0_fits.json")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same directory as JSON)")
    args = parser.parse_args()

    runs, data = load(args.json)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for half, title in HALVES.items():
        out = out_dir / f"z0_fits_vs_run_{half}.pdf"
        plot_half(half, title, runs, data, out)


if __name__ == "__main__":
    main()
