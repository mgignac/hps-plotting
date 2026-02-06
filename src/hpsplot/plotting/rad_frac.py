"""Radiative fraction plots: ratio of numerator sample(s) to sum of denominator sample(s)."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import add_hps_label

logger = logging.getLogger(__name__)


def plot_rad_frac(plot_cfg, hist_cfg, region_cfg, results, samples_map, output_dir,
                  output_format="pdf"):
    """Create a radiative fraction plot.

    Computes the bin-by-bin ratio of numerator sample(s) to the sum of all
    denominator sample(s), with Gaussian error propagation.

    Parameters
    ----------
    plot_cfg : PlotConfig
    hist_cfg : HistogramConfig
    region_cfg : RegionConfig
    results : dict
        results[region_name][sample_name][hist_name] -> HistogramData
    samples_map : dict
        sample_name -> SampleConfig
    output_dir : str
    output_format : str
    """
    region_name = region_cfg.name
    hist_name = hist_cfg.name
    region_results = results.get(region_name, {})

    # Classify samples by their rad_frac_role
    num_names = []
    den_names = []
    for s_name in plot_cfg.samples:
        s_cfg = samples_map.get(s_name)
        if s_cfg is None:
            continue
        if s_cfg.rad_frac_role == "numerator":
            num_names.append(s_name)
        elif s_cfg.rad_frac_role == "denominator":
            den_names.append(s_name)

    if not num_names:
        logger.warning("rad_frac plot '%s': no samples with rad_frac_role='numerator'",
                        plot_cfg.name)
        return
    if not den_names:
        logger.warning("rad_frac plot '%s': no samples with rad_frac_role='denominator'",
                        plot_cfg.name)
        return

    # Sum numerator histograms
    bin_edges = None
    num_contents = None
    num_errors_sq = None

    for s_name in num_names:
        hdata = region_results.get(s_name, {}).get(hist_name)
        if hdata is None:
            continue
        if bin_edges is None:
            bin_edges = hdata.bin_edges
            num_contents = hdata.bin_contents.copy()
            num_errors_sq = hdata.bin_errors ** 2
        else:
            num_contents += hdata.bin_contents
            num_errors_sq += hdata.bin_errors ** 2

    # Sum denominator histograms
    den_contents = None
    den_errors_sq = None

    for s_name in den_names:
        hdata = region_results.get(s_name, {}).get(hist_name)
        if hdata is None:
            continue
        if bin_edges is None:
            bin_edges = hdata.bin_edges
        if den_contents is None:
            den_contents = hdata.bin_contents.copy()
            den_errors_sq = hdata.bin_errors ** 2
        else:
            den_contents += hdata.bin_contents
            den_errors_sq += hdata.bin_errors ** 2

    if num_contents is None or den_contents is None or bin_edges is None:
        logger.warning("No data for rad_frac plot %s / %s / %s",
                        plot_cfg.name, region_name, hist_name)
        return

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute ratio and propagate errors: sigma_r = r * sqrt((sN/N)^2 + (sD/D)^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den_contents > 0, num_contents / den_contents, 0.0)
        rel_num = np.where(num_contents > 0, num_errors_sq / num_contents ** 2, 0.0)
        rel_den = np.where(den_contents > 0, den_errors_sq / den_contents ** 2, 0.0)
        ratio_err = ratio * np.sqrt(rel_num + rel_den)

    # Build labels
    num_labels = [samples_map[n].label for n in num_names]
    den_labels = [samples_map[n].label for n in den_names]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        bin_centers, ratio,
        yerr=ratio_err,
        fmt="o",
        color="black",
        markersize=4,
        linewidth=1.2,
    )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel(hist_cfg.x_label if hist_cfg.x_label else hist_cfg.variable)
    frac_label = f"{' + '.join(num_labels)} / ({' + '.join(den_labels)})"
    ax.set_ylabel(f"Radiative fraction")
    ax.set_ylim(plot_cfg.ratio_y_min, plot_cfg.ratio_y_max)

    if hist_cfg.log_y:
        ax.set_yscale("log")

    ax.legend([frac_label], loc="upper right")
    add_hps_label(ax)

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{plot_cfg.name}_{region_name}_{hist_name}.{output_format}"
    outpath = outdir / fname
    fig.savefig(outpath)
    plt.close(fig)
    logger.info("Saved: %s", outpath)
