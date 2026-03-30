"""Binned comparison plot: overlay normalised distributions in bins of a second variable.

Produces a row of panels — one per bin — each showing all samples normalised to
unit area with a ratio panel below (relative to the first sample).

A second figure shows the fitted Gaussian mean of the distribution vs |bin_variable|
for each sample, with a linear (ax+b) fit overlaid.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .style import add_hps_label
from .smearing import _normalise_to_unity   # reuse from smearing
from ..region import Region
from ..sample import Sample
from ..utils import safe_evaluate, extract_branch_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _fit_gaussian_mean(centers, contents):
    """Fit a Gaussian to a histogram; return (mean, mean_err).

    Returns (nan, nan) on failure.
    """
    mask = contents > 0
    if mask.sum() < 4:
        return np.nan, np.nan
    total = contents[mask].sum()
    if total == 0:
        return np.nan, np.nan

    mu0 = np.average(centers[mask], weights=contents[mask])
    var0 = np.average((centers[mask] - mu0) ** 2, weights=contents[mask])
    sig0 = max(np.sqrt(var0), 1e-4)
    amp0 = contents.max()

    try:
        popt, pcov = curve_fit(
            _gaussian, centers[mask], contents[mask],
            p0=[amp0, mu0, sig0],
            maxfev=10_000,
        )
        mean = popt[1]
        mean_err = np.sqrt(abs(pcov[1, 1]))
        return float(mean), float(mean_err)
    except Exception:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_binned_comparison(plot_cfg, region_cfg, config, samples_map,
                           output_dir, output_format):
    """Plot normalised distributions of ``variable`` in bins of ``bin_variable``.

    Layout
    ------
    Figure 1 — panel plot:
      One column per bin.  Each column contains:
        • top panel  — normalised histogram overlay for all samples
        • bottom panel — ratio relative to the first sample

    Figure 2 — mean summary plot:
      Fitted Gaussian mean of ``variable`` vs |bin_variable| bin centre,
      one series per sample, overlaid with a linear ax+b fit.
    """
    bcfg = plot_cfg.binned
    region = Region(region_cfg) if region_cfg is not None else None

    bin_edges = np.array(bcfg.bin_edges, dtype=float)
    if len(bin_edges) < 2:
        logger.error("binned plot '%s': need at least 2 bin_edges.", plot_cfg.name)
        return

    n_bins = len(bin_edges) - 1
    hist_edges = np.linspace(bcfg.x_min, bcfg.x_max, bcfg.bins + 1)
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    # Bin centres (signed and absolute)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers_abs = np.abs(bin_centers)

    # --- load data for each sample ------------------------------------------
    sample_data = {}   # name → (var_arr, bin_arr)
    for name in plot_cfg.samples:
        cfg_s = samples_map[name]
        ea = {**config.aliases, **cfg_s.aliases}

        branches = set()
        branches |= extract_branch_names(bcfg.variable)
        branches |= extract_branch_names(bcfg.bin_variable)
        if region is not None:
            branches |= extract_branch_names(region.selection)
        if cfg_s.selection:
            branches |= extract_branch_names(cfg_s.selection)
        if bcfg.selection:
            branches |= extract_branch_names(bcfg.selection)

        sample = Sample(cfg_s)
        data = sample.load(branches, aliases=ea)

        mask = np.ones(len(next(iter(data.values()))), dtype=bool)
        if region is not None:
            mask &= region.apply(data)
        if cfg_s.selection:
            mask &= np.asarray(safe_evaluate(cfg_s.selection, data), dtype=bool)
        if bcfg.selection:
            mask &= np.asarray(safe_evaluate(bcfg.selection, data), dtype=bool)

        var_arr = np.asarray(safe_evaluate(bcfg.variable,     data, mask=mask), dtype=float)
        bin_arr = np.asarray(safe_evaluate(bcfg.bin_variable, data, mask=mask), dtype=float)
        sample_data[name] = (var_arr, bin_arr)
        logger.info("Loaded '%s': %d events after selection", name, int(mask.sum()))

    # --- Figure 1: panel plot -----------------------------------------------
    fig_w = max(3.5 * n_bins, 8)
    fig, axes = plt.subplots(
        2, n_bins,
        figsize=(fig_w, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex="col",
    )
    if n_bins == 1:
        axes = np.array(axes).reshape(2, 1)

    fig.subplots_adjust(hspace=0.05, wspace=0.3)

    region_label = region_cfg.label if region_cfg is not None else ""
    extra_lines = [l for l in [bcfg.label, region_label] if l]
    add_hps_label(axes[0, 0], lumi=config.luminosity, extra_lines=extra_lines)

    ref_name = plot_cfg.samples[0]

    # means_dict: name → (list of mean, list of mean_err)  indexed by bin col
    means_dict = {name: ([], []) for name in plot_cfg.samples}

    for col, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        bin_title = (f"${bcfg.bin_label or bcfg.bin_variable}$"
                     f" $\\in$ [{lo:.3g}, {hi:.3g}]")
        ax_top.set_title(bin_title, fontsize=7)

        ref_contents = None

        for i, name in enumerate(plot_cfg.samples):
            cfg_s = samples_map[name]
            var_arr, bin_arr = sample_data[name]

            sel = (bin_arr >= lo) & (bin_arr < hi)
            contents, _ = np.histogram(var_arr[sel], bins=hist_edges)
            errors = np.sqrt(contents).astype(float)
            contents = contents.astype(float)

            # Fit Gaussian mean before normalising (raw counts better for fit)
            mean, mean_err = _fit_gaussian_mean(centers, contents)
            means_dict[name][0].append(mean)
            means_dict[name][1].append(mean_err)

            if bcfg.normalize:
                contents, errors = _normalise_to_unity(centers, contents, errors)

            ax_top.errorbar(
                centers, contents, yerr=errors,
                fmt="o-", color=cfg_s.color, label=cfg_s.label,
                markersize=2, lw=1,
            )

            # Annotate fitted mean on top panel
            if np.isfinite(mean):
                ax_top.axvline(mean, color=cfg_s.color, lw=1, ls="--", alpha=0.7)

            if i == 0:
                ref_contents = contents.copy()
            else:
                valid = ref_contents > 0
                ratio = np.where(valid, contents / ref_contents, np.nan)
                ax_bot.errorbar(
                    centers, ratio,
                    fmt="o-", color=cfg_s.color, markersize=2, lw=1,
                )

        ax_bot.axhline(1.0, color="gray", lw=1, ls="--")
        ax_bot.set_ylim(bcfg.ratio_y_min, bcfg.ratio_y_max)
        ax_bot.set_xlabel(bcfg.x_label or bcfg.variable, fontsize=8)

        if col == 0:
            y_label = ("1/N dN/d(" + (bcfg.x_label or bcfg.variable) + ")"
                       if bcfg.normalize else "Events")
            ax_top.set_ylabel(y_label, fontsize=8)
            ax_bot.set_ylabel(f"/ {samples_map[ref_name].label}", fontsize=8)

        if bcfg.log_y:
            ax_top.set_yscale("log")

    axes[0, 0].legend(fontsize=8, loc="upper right")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    region_tag = f"_{region_cfg.name}" if region_cfg is not None else ""

    fname = f"{plot_cfg.name}{region_tag}.{output_format}"
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outdir / fname)

    # --- Figure 2: mean z0 vs |tanL| ----------------------------------------
    _plot_mean_vs_abs_binvar(
        bin_centers, bin_centers_abs, means_dict, samples_map, plot_cfg, bcfg,
        config, outdir, output_format, region_tag, region_label,
    )


def _linear_fit_and_plot(ax, x, y, yerr, color, label):
    """Weighted linear fit ax+b; draw on ax.

    Returns
    -------
    annotation : str
    a : float or None   slope
    b : float or None   intercept
    """
    valid = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    ax.errorbar(
        x[valid], y[valid], yerr=yerr[valid],
        fmt="o", color=color, label=label,
        markersize=5, lw=1.5, capsize=3,
    )
    if valid.sum() < 2:
        logger.info("  [%s] insufficient points for linear fit", label)
        return f"{label}: insufficient points", None, None
    xv, yv, wv = x[valid], y[valid], 1.0 / yerr[valid] ** 2
    coeffs = np.polyfit(xv, yv, 1, w=wv)
    a, b = float(coeffs[0]), float(coeffs[1])
    logger.info("  [%s] linear fit: a = %+.6f,  b = %+.6f  (mean_z0 = a*|tanL| + b)", label, a, b)
    x_fit = np.linspace(xv.min(), xv.max(), 200)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, ls="--", lw=1.2)
    sign_b = "+" if b >= 0 else "-"
    return f"{label}: $a={a:.4f}$, $b\\,{sign_b}\\,{abs(b):.4f}$", a, b


def _plot_mean_vs_abs_binvar(bin_centers, bin_centers_abs, means_dict,
                              samples_map, plot_cfg, bcfg, config,
                              outdir, output_format, region_tag, region_label):
    """Two-panel figure: mean z0 vs |tanL| separated by tanL > 0 (top) and tanL < 0 (bottom).

    Each panel has a weighted linear ax+b fit per sample with parameters shown.
    """
    bin_label_str = bcfg.bin_label or bcfg.bin_variable
    x_label_str   = bcfg.x_label or bcfg.variable

    # Masks for top (tanL > 0) and bottom (tanL < 0) halves
    top_mask = bin_centers > 0
    bot_mask = bin_centers < 0

    # Skip bins exactly at zero for both halves
    panel_specs = [
        (top_mask, r"$\tan\lambda > 0$ (top detector)", "top"),
        (bot_mask, r"$\tan\lambda < 0$ (bottom detector)", "bot"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    fit_results = {}   # accumulated for JSON output: {key: {slope, intercept}}

    for ax, (pmask, panel_title, half_key) in zip(axes, panel_specs):
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel(f"${bin_label_str}$", fontsize=10)
        ax.set_ylabel(f"Mean {x_label_str}", fontsize=10)

        logger.info("Mean z0 linear fit — %s:", panel_title.replace("$", ""))
        text_lines = []
        all_y, all_yerr = [], []
        for name in plot_cfg.samples:
            cfg_s = samples_map[name]
            x    = bin_centers[pmask]
            y    = np.array(means_dict[name][0])[pmask]
            yerr = np.array(means_dict[name][1])[pmask]
            ann, a, b = _linear_fit_and_plot(ax, x, y, yerr, cfg_s.color, cfg_s.label)
            text_lines.append(ann)
            if a is not None and bcfg.json_key_prefix:
                key = f"{bcfg.json_key_prefix}_{half_key}"
                fit_results[key] = {"slope": a, "intercept": b}
            valid = np.isfinite(y) & np.isfinite(yerr)
            all_y.extend(y[valid])
            all_yerr.extend(yerr[valid])

        # Dynamic y-axis: pad by 48% of data range; enforce minimum span of ±0.02
        if all_y:
            lo_pts = np.array(all_y) - np.array(all_yerr)
            hi_pts = np.array(all_y) + np.array(all_yerr)
            pad = 0.48 * (hi_pts.max() - lo_pts.min() or 1e-4)
            y_lo = min(lo_pts.min() - pad, -0.02)
            y_hi = max(hi_pts.max() + pad,  0.02)
            ax.set_ylim(y_lo, y_hi)

        ax.axhline(0.0, color="gray", lw=0.8, ls=":")

        # Place legend and fit-text on opposite corners to avoid overlap
        leg = ax.legend(fontsize=9, loc="upper right")
        fit_text = "\n".join(text_lines)
        ax.text(
            0.96, 0.04, fit_text,
            transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

    extra_lines = [l for l in [bcfg.label, region_label] if l]
    add_hps_label(axes[0], lumi=config.luminosity, extra_lines=extra_lines)

    fname = f"{plot_cfg.name}{region_tag}_mean_vs_tanl.{output_format}"
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outdir / fname)

    # --- Write fit results to JSON ------------------------------------------
    if bcfg.json_output and fit_results:
        run_key = config.run_label or "default"
        json_path = Path(bcfg.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
        if run_key not in existing:
            existing[run_key] = {}
        existing[run_key].update(fit_results)
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info("JSON fit results written: %s (run '%s')", json_path, run_key)
