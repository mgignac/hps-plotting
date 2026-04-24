"""Normalized shape comparison overlay plots."""

import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from .fit import fit_histogram
from .style import add_hps_label

logger = logging.getLogger(__name__)


def plot_overlay(plot_cfg, hist_cfg, region_cfg, results, samples_map, output_dir,
                 output_format="pdf", run_label=""):
    """Create an overlay plot comparing histogram shapes across samples.

    When plot_cfg.data_sample is set, the data sample is drawn as black markers
    and a Data/MC ratio panel is added below the main panel.

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
    data_name = plot_cfg.data_sample

    region_results = results.get(region_name, {})

    # Determine if we draw a ratio panel
    has_ratio = bool(data_name) and data_name in region_results

    if has_ratio:
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax_ratio = None

    # Collect normalized data histogram for ratio
    data_hdata_norm = None
    if has_ratio:
        raw = region_results.get(data_name, {}).get(hist_name)
        if raw is not None:
            data_hdata_norm = raw.normalized() if normalize else raw

    mc_samples = [s for s in plot_cfg.samples if s != data_name]

    for s_name in plot_cfg.samples:
        hdata = region_results.get(s_name, {}).get(hist_name)
        if hdata is None:
            continue

        s_cfg = samples_map[s_name]
        label = f"{s_cfg.label} ({hdata.integral:.0f})"
        hplot = hdata.normalized() if normalize else hdata
        bin_centers = hplot.bin_centers

        is_data = s_name == data_name
        if is_data:
            ax.errorbar(
                bin_centers, hplot.bin_contents,
                yerr=hplot.bin_errors,
                fmt="o", color=s_cfg.color,
                label=label,
                markersize=4, linewidth=1.0,
                zorder=5,
            )
        else:
            ax.step(
                bin_centers, hplot.bin_contents,
                where="mid",
                color=s_cfg.color,
                label=label,
                linewidth=1.5,
            )
            ax.fill_between(
                bin_centers,
                hplot.bin_contents - hplot.bin_errors,
                hplot.bin_contents + hplot.bin_errors,
                step="mid",
                color=s_cfg.color,
                alpha=0.15,
            )

        # Ratio panel
        if ax_ratio is not None and not is_data and data_hdata_norm is not None:
            hmc_norm = hdata.normalized() if normalize else hdata
            denom = hmc_norm.bin_contents
            numer = data_hdata_norm.bin_contents
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(denom > 0, numer / denom, np.nan)
                ratio_err = np.where(
                    denom > 0,
                    data_hdata_norm.bin_errors / denom,
                    np.nan,
                )
            ax_ratio.errorbar(
                bin_centers, ratio,
                yerr=ratio_err,
                fmt="o", color=s_cfg.color,
                markersize=4, linewidth=1.0,
            )

    ax.set_ylabel("Normalized" if normalize else (hist_cfg.y_label if hist_cfg.y_label else "Events"))

    if hist_cfg.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-4 if normalize else 0.1)
    else:
        ax.set_ylim(bottom=0)

    if hist_cfg.y_top_scale != 1.0:
        ax.set_ylim(top=ax.get_ylim()[1] * hist_cfg.y_top_scale)

    # Fit overlay
    if plot_cfg.fit is not None:
        fit_cfg = plot_cfg.fit
        fit_sample = fit_cfg.sample or plot_cfg.samples[0]
        hdata = region_results.get(fit_sample, {}).get(hist_name)
        if hdata is not None:
            hfit = hdata.normalized() if normalize else hdata
            result = fit_histogram(hfit, fit_cfg, hist_cfg)
            if result is not None:
                popt, pcov, x_curve, y_curve, param_labels = result
                perr = np.sqrt(np.diag(pcov))
                ax.plot(x_curve, y_curve, color=fit_cfg.color, linewidth=2,
                        linestyle="--", label="Fit")
                if fit_cfg.show_params:
                    lines = []
                    for name, val, err in zip(param_labels, popt, perr):
                        lines.append(f"${name} = {val:.3g} \\pm {err:.2g}$")
                    ax.text(0.95, 0.55, "\n".join(lines),
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment="top", horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend(loc="upper right")
    add_hps_label(ax, run_label=run_label)

    if ax_ratio is not None:
        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_ratio.set_ylabel("Data / MC")
        ax_ratio.set_xlabel(hist_cfg.x_label if hist_cfg.x_label else hist_cfg.variable)
        ax_ratio.set_ylim(plot_cfg.ratio_y_min, plot_cfg.ratio_y_max)
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel(hist_cfg.x_label if hist_cfg.x_label else hist_cfg.variable)

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{plot_cfg.name}_{region_name}_{hist_name}.{output_format}"
    outpath = outdir / fname
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outpath)
