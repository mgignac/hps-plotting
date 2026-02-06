"""Stacked MC + data + ratio panel plots."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .style import add_hps_label

logger = logging.getLogger(__name__)


def plot_stack(plot_cfg, hist_cfg, region_cfg, results, samples_map, output_dir,
               output_format="pdf"):
    """Create a stacked histogram plot with data overlay and ratio panel.

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

    # Separate backgrounds, signal, and data
    bg_names = []
    sig_names = list(plot_cfg.signal_samples)
    data_name = plot_cfg.data_sample

    for s_name in plot_cfg.samples:
        if s_name == data_name:
            continue
        if s_name in sig_names:
            continue
        s_cfg = samples_map.get(s_name)
        if s_cfg and s_cfg.sample_type == "signal":
            sig_names.append(s_name)
        elif s_cfg and s_cfg.sample_type == "background":
            bg_names.append(s_name)

    # Get histogram data
    region_results = results.get(region_name, {})

    # Build background stack
    bg_contents = []
    bg_errors = []
    bg_colors = []
    bg_labels = []
    bin_edges = None

    for s_name in bg_names:
        hdata = region_results.get(s_name, {}).get(hist_name)
        if hdata is None:
            continue
        bg_contents.append(hdata.bin_contents.copy())
        bg_errors.append(hdata.bin_errors.copy())
        bg_colors.append(samples_map[s_name].color)
        bg_labels.append(samples_map[s_name].label)
        bin_edges = hdata.bin_edges

    if bin_edges is None:
        # Try to get edges from any available histogram
        for s_name in plot_cfg.samples:
            hdata = region_results.get(s_name, {}).get(hist_name)
            if hdata is not None:
                bin_edges = hdata.bin_edges
                break
    if bin_edges is None:
        logger.warning("No data available for %s / %s / %s", region_name, hist_name,
                       plot_cfg.name)
        return

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    has_data = data_name and data_name in region_results and hist_name in region_results.get(data_name, {})
    has_ratio = has_data and len(bg_contents) > 0

    # Normalize backgrounds to data yield using user-supplied fractions
    if plot_cfg.normalize_to_data and has_data and bg_contents:
        data_yield = np.sum(region_results[data_name][hist_name].bin_contents)
        fracs = plot_cfg.background_fractions

        for i, s_name in enumerate(bg_names):
            hdata = region_results.get(s_name, {}).get(hist_name)
            if hdata is None:
                continue
            frac = fracs.get(s_name, 0.0)
            target = data_yield * frac
            current = np.sum(bg_contents[i])
            if current > 0:
                scale = target / current
                bg_contents[i] = bg_contents[i] * scale
                bg_errors[i] = hdata.bin_errors * scale
            else:
                bg_errors[i] = hdata.bin_errors.copy()

    # Create figure
    if has_ratio:
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
    else:
        fig, ax_main = plt.subplots(figsize=(8, 6))
        ax_ratio = None

    # Plot stacked backgrounds
    if bg_contents:
        bg_stack = np.array(bg_contents)
        ax_main.hist(
            [bin_centers] * len(bg_contents),
            bins=bin_edges,
            weights=bg_contents,
            stacked=True,
            histtype="stepfilled",
            color=bg_colors,
            label=bg_labels,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # MC total and uncertainty band
        mc_total = np.sum(bg_stack, axis=0)
        mc_errors_sq = np.zeros_like(mc_total)
        for errs in bg_errors:
            mc_errors_sq += errs ** 2
        mc_errors = np.sqrt(mc_errors_sq)

        # Hatched error band
        ax_main.fill_between(
            bin_centers,
            mc_total - mc_errors,
            mc_total + mc_errors,
            step="mid",
            hatch="///",
            facecolor="none",
            edgecolor="gray",
            linewidth=0,
            label="MC stat. unc.",
        )
    else:
        mc_total = None
        mc_errors = None

    # Plot signal overlays (dashed step), optionally scaled to data yield
    for s_name in sig_names:
        hdata = region_results.get(s_name, {}).get(hist_name)
        if hdata is None:
            continue
        contents = hdata.bin_contents
        sig_label = samples_map[s_name].label

        if plot_cfg.normalize_to_data and has_data and s_name in plot_cfg.signal_fractions:
            data_yield = np.sum(region_results[data_name][hist_name].bin_contents)
            frac = plot_cfg.signal_fractions[s_name]
            current = np.sum(contents)
            if current > 0:
                scale = (data_yield * frac) / current
                contents = contents * scale
                sig_label = f"{sig_label} (x{scale:.1f})"

        ax_main.step(
            bin_centers, contents,
            where="mid",
            color=samples_map[s_name].color,
            label=sig_label,
            linestyle="--",
            linewidth=2,
        )

    # Plot data points
    if has_data:
        data_hist = region_results[data_name][hist_name]
        ax_main.errorbar(
            bin_centers,
            data_hist.bin_contents,
            yerr=data_hist.bin_errors,
            fmt="o",
            color=samples_map[data_name].color,
            label=samples_map[data_name].label,
            markersize=4,
            linewidth=1.2,
            zorder=5,
        )

    # Main panel formatting
    ax_main.set_ylabel(hist_cfg.y_label if hist_cfg.y_label else "Events")
    if hist_cfg.log_y:
        ax_main.set_yscale("log")
        ax_main.set_ylim(bottom=0.1)
    else:
        ax_main.set_ylim(bottom=0)
    ax_main.legend(loc="upper right")
    add_hps_label(ax_main)

    if has_ratio:
        plt.setp(ax_main.get_xticklabels(), visible=False)
    else:
        ax_main.set_xlabel(hist_cfg.x_label if hist_cfg.x_label else hist_cfg.variable)

    # Ratio panel
    if has_ratio and mc_total is not None:
        data_hist = region_results[data_name][hist_name]

        # Data / MC ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mc_total > 0, data_hist.bin_contents / mc_total, 0)
            ratio_err = np.where(mc_total > 0, data_hist.bin_errors / mc_total, 0)

        ax_ratio.errorbar(
            bin_centers, ratio,
            yerr=ratio_err,
            fmt="o",
            color="black",
            markersize=4,
            linewidth=1.2,
        )

        # MC stat error band on ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            mc_ratio_err = np.where(mc_total > 0, mc_errors / mc_total, 0)

        ax_ratio.fill_between(
            bin_centers,
            1 - mc_ratio_err,
            1 + mc_ratio_err,
            step="mid",
            color="gray",
            alpha=0.3,
        )

        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_ratio.set_ylabel("Data / MC")
        ax_ratio.set_xlabel(hist_cfg.x_label if hist_cfg.x_label else hist_cfg.variable)
        ax_ratio.set_ylim(plot_cfg.ratio_y_min, plot_cfg.ratio_y_max)

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{plot_cfg.name}_{region_name}_{hist_name}.{output_format}"
    outpath = outdir / fname
    fig.savefig(outpath)
    plt.close(fig)
    logger.info("Saved: %s", outpath)
