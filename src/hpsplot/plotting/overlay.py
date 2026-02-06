"""Normalized shape comparison overlay plots."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import add_hps_label

logger = logging.getLogger(__name__)


def plot_overlay(plot_cfg, hist_cfg, region_cfg, results, samples_map, output_dir,
                 output_format="pdf"):
    """Create an overlay plot comparing histogram shapes across samples.

    Parameters
    ----------
    plot_cfg : PlotConfig
    hist_cfg : HistogramConfig
    region_cfg : RegionConfig
    results : dict
        results[region_name][sample_name][hist_name] → HistogramData
    samples_map : dict
        sample_name → SampleConfig
    output_dir : str
    output_format : str
    """
    region_name = region_cfg.name
    hist_name = hist_cfg.name
    normalize = plot_cfg.normalize

    region_results = results.get(region_name, {})

    fig, ax = plt.subplots(figsize=(8, 6))

    for s_name in plot_cfg.samples:
        hdata = region_results.get(s_name, {}).get(hist_name)
        if hdata is None:
            continue

        if normalize:
            hdata = hdata.normalized()

        s_cfg = samples_map[s_name]
        bin_centers = hdata.bin_centers

        ax.step(
            bin_centers, hdata.bin_contents,
            where="mid",
            color=s_cfg.color,
            label=s_cfg.label,
            linewidth=1.5,
        )

        # Error band
        ax.fill_between(
            bin_centers,
            hdata.bin_contents - hdata.bin_errors,
            hdata.bin_contents + hdata.bin_errors,
            step="mid",
            color=s_cfg.color,
            alpha=0.15,
        )

    ax.set_xlabel(hist_cfg.x_label if hist_cfg.x_label else hist_cfg.variable)
    ylabel = "Normalized" if normalize else (hist_cfg.y_label if hist_cfg.y_label else "Events")
    ax.set_ylabel(ylabel)

    if hist_cfg.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1 if not normalize else 1e-4)
    else:
        ax.set_ylim(bottom=0)

    ax.legend(loc="upper right")
    add_hps_label(ax)

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{plot_cfg.name}_{region_name}_{hist_name}.{output_format}"
    outpath = outdir / fname
    fig.savefig(outpath)
    plt.close(fig)
    logger.info("Saved: %s", outpath)
