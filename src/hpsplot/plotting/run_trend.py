"""Integrated yield vs run number plots for --per-file pipeline."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from .style import add_hps_label

logger = logging.getLogger(__name__)


def plot_run_trend(plot_cfg, run_records, region_name, samples_map, output_dir, output_format="pdf"):
    """Plot integrated event yields vs run number with a data/MC ratio panel.

    Parameters
    ----------
    plot_cfg : PlotConfig
        The run_trend plot configuration.
    run_records : list of (run_label, dict, float)
        Each entry is ``(run_label, {sample_name: yield}, lumi)``,
        one per run in the order they were processed.
    region_name : str
        Name of the region these yields correspond to.
    samples_map : dict
        Mapping sample_name → SampleConfig.
    output_dir : str
    output_format : str
    """
    if not run_records:
        logger.warning("No run data collected for run_trend plot '%s' / '%s'", plot_cfg.name, region_name)
        return

    run_labels = np.array([r[0] for r in run_records])
    yield_dicts = [r[1] for r in run_records]
    lumis = np.array([r[2] for r in run_records])

    data_name = plot_cfg.data_sample
    bg_names = []
    for s_name in plot_cfg.samples:
        if s_name == data_name:
            continue
        s_cfg = samples_map.get(s_name)
        if s_cfg and s_cfg.sample_type == "background":
            bg_names.append(s_name)

    n_runs = len(run_labels)

    # Sequential bin positions — every run gets its own labeled bin.
    x_pos = np.arange(n_runs)
    bar_width = 0.8
    x_label = "Run number"
    # Font size for run-number tick labels: shrink for large run counts
    tick_fontsize = max(2, min(7, int(400 / max(n_runs, 1))))

    # Collect yields; normalise to events / pb^-1 where lumi > 0
    lumi_safe = np.where(lumis > 0, lumis, np.nan)
    data_yields_raw = np.array([d.get(data_name, 0.0) for d in yield_dicts])
    data_yields = data_yields_raw / lumi_safe
    bg_by_sample_raw = {s: np.array([d.get(s, 0.0) for d in yield_dicts]) for s in bg_names}
    bg_by_sample = {s: bg_by_sample_raw[s] / lumi_safe for s in bg_names}
    mc_total = np.zeros(n_runs)
    for s in bg_names:
        mc_total += np.nan_to_num(bg_by_sample[s])

    has_data = bool(data_name) and np.any(np.nan_to_num(data_yields) > 0)
    has_mc = np.any(mc_total > 0)
    has_ratio = has_data and has_mc

    fig_width = 12
    if has_ratio:
        fig = plt.figure(figsize=(fig_width, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)
    else:
        fig, ax_main = plt.subplots(figsize=(fig_width, 6))
        ax_ratio = None

    # Stacked MC bars
    if has_mc and bg_names:
        bottom = np.zeros(n_runs)
        for s_name in bg_names:
            heights = bg_by_sample[s_name]
            s_cfg = samples_map.get(s_name)
            ax_main.bar(
                x_pos, heights, bottom=bottom,
                label=f"{s_cfg.label} ({np.nansum(bg_by_sample_raw[s_name]):.0f} total)" if s_cfg else s_name,
                color=s_cfg.color if s_cfg else None,
                alpha=0.7,
                width=bar_width,
            )
            bottom += heights

    # Data points with Poisson error bars (sqrt(N_raw) / lumi)
    if has_data:
        data_err = np.sqrt(np.maximum(data_yields_raw, 1.0)) / lumi_safe
        s_cfg = samples_map.get(data_name)
        ax_main.errorbar(
            x_pos, data_yields, yerr=data_err,
            fmt="o",
            color=s_cfg.color if s_cfg else "black",
            label=f"{s_cfg.label} ({np.sum(data_yields_raw):.0f} total)" if s_cfg else f"Data ({np.sum(data_yields_raw):.0f} total)",
            markersize=5,
            linewidth=1.2,
            zorder=5,
        )

    ax_main.set_ylabel(r"Events / pb$^{-1}$")
    ax_main.legend(loc="upper right")
    add_hps_label(ax_main)

    def _apply_run_ticks(ax, visible=True):
        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_labels, rotation=90, ha="center", fontsize=tick_fontsize,
                           visible=visible)

    if not has_ratio:
        ax_main.set_xlabel(x_label)
        _apply_run_ticks(ax_main)

    # Ratio panel
    if has_ratio and ax_ratio is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mc_total > 0, np.nan_to_num(data_yields) / mc_total, np.nan)
            ratio_err = np.where(mc_total > 0, data_err / mc_total, np.nan)

        ax_ratio.errorbar(
            x_pos, ratio, yerr=ratio_err,
            fmt="o", color="black", markersize=4, linewidth=1.2,
        )
        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_ratio.set_ylabel("Data / MC")
        ax_ratio.set_ylim(plot_cfg.ratio_y_min, plot_cfg.ratio_y_max)
        ax_ratio.set_xlabel(x_label)
        _apply_run_ticks(ax_ratio)
        # Hide labels on top canvas after setting them on the shared x-axis
        plt.setp(ax_main.get_xticklabels(), visible=False)

    # Save plot
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{plot_cfg.name}_{region_name}"
    outpath = outdir / f"{stem}.{output_format}"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outpath)

    # Write JSON summary: one entry per run
    json_data = {}
    for i, run_label in enumerate(run_labels):
        entry = {
            "lumi": float(lumis[i]),
            "data": float(data_yields[i]),
            "mc_total": float(mc_total[i]),
            "backgrounds": {s: float(bg_by_sample[s][i]) for s in bg_names},
        }
        if has_ratio and mc_total[i] > 0:
            entry["data_mc_ratio"] = float(data_yields[i] / mc_total[i])
        json_data[run_label] = entry

    json_path = outdir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("Saved: %s", json_path)
