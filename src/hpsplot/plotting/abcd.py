"""ABCD background estimation method.

Scans over invariant mass bins and estimates background using:
    N_A(bkg) = N_B * N_C / N_D

Regions (defined per mass center):
    A: signal mass window  AND signal Z region    (observed signal-region count)
    B: signal mass window  AND control Z region
    C: sideband mass       AND signal Z region
    D: sideband mass       AND control Z region
"""

import logging
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import add_hps_label
from ..region import Region
from ..sample import Sample
from ..utils import safe_evaluate, extract_branch_names

logger = logging.getLogger(__name__)


def estimate_background(n_b, n_c, n_d):
    """Compute ABCD background estimate N_B*N_C/N_D with Poisson error propagation.

    Returns (estimate, error). Returns (0, 0) if N_D == 0.
    """
    if n_d == 0:
        return 0.0, 0.0
    est = n_b * n_c / n_d
    if est > 0 and n_b > 0 and n_c > 0:
        err = est * sqrt(1.0 / n_b + 1.0 / n_c + 1.0 / n_d)
    else:
        err = 0.0
    return est, err


def _compute_window(abcd_cfg, region_name, mass_center):
    """Return (hw, gap, sw) for a given mass center.

    If abcd_cfg.mass_resolution has an entry for region_name, the window is
    computed from sigma(m) evaluated at mass_center:
        hw  = sr_sigmas  * sigma(m)
        gap = gap_sigmas * sigma(m)
        sw  = sb_sigmas  * sigma(m)
    Otherwise the fixed abcd_cfg fields are used.
    """
    res_expr = abcd_cfg.mass_resolution.get(region_name)
    if not res_expr:
        return (abcd_cfg.mass_window_half_width,
                abcd_cfg.sideband_gap,
                abcd_cfg.sideband_width)
    sigma = float(safe_evaluate(res_expr, {"m": np.array([mass_center])}))
    return (abcd_cfg.sr_sigmas  * sigma,
            abcd_cfg.gap_sigmas * sigma,
            abcd_cfg.sb_sigmas  * sigma)


def _count_abcd(mass_arr, z_arr, mass_center, hw, gap, sw, abcd_cfg, weights=None):
    """Count weighted events in ABCD regions for a given mass center.

    Returns dict with keys 'A', 'B', 'C', 'D', each a (sum, sum_err) tuple.
    When weights is None, uses unweighted counts with sqrt(N) Poisson errors.
    hw, gap, sw are the pre-computed window parameters (GeV).
    """

    mass_sig = (mass_arr >= mass_center - hw) & (mass_arr <= mass_center + hw)
    mass_sb = (
        ((mass_arr >= mass_center - hw - gap - sw) & (mass_arr <= mass_center - hw - gap)) |
        ((mass_arr >= mass_center + hw + gap)       & (mass_arr <= mass_center + hw + gap + sw))
    )
    z_sig  = (z_arr >= abcd_cfg.z_signal_min)  & (z_arr <= abcd_cfg.z_signal_max)
    z_ctrl = (z_arr >= abcd_cfg.z_control_min) & (z_arr <= abcd_cfg.z_control_max)

    def _sum(mask):
        if weights is None:
            n = float(np.sum(mask))
            return n, sqrt(n) if n > 0 else 0.0
        w = weights[mask]
        s = float(np.sum(w))
        e = float(np.sqrt(np.sum(w ** 2))) if len(w) > 0 else 0.0
        return s, e

    return {
        "A": _sum(mass_sig & z_sig),
        "B": _sum(mass_sig & z_ctrl),
        "C": _sum(mass_sb  & z_sig),
        "D": _sum(mass_sb  & z_ctrl),
    }


def _compute_sample_counts(sample_name, sample_cfg, mass_centers, abcd_cfg,
                           region, effective_aliases, luminosity):
    """Load a sample and return per-mass-bin ABCD region counts.

    Returns dict:
        na, na_err, nb, nc, nd  — numpy arrays of length len(mass_centers)
    """
    sample = Sample(sample_cfg)

    # Resolve lumi weight (same logic / log format as results.py)
    lumi_weight = 1.0
    if sample_cfg.lumi_scale:
        lumi_weight, xsec, n_gen = sample.get_lumi_weight(luminosity)
        total_scale = sample.scale * lumi_weight
        logger.info(
            "Lumi weight for '%s':\n"
            "    cross_section  = %.4g\n"
            "    luminosity     = %.4g\n"
            "    n_generated    = %d\n"
            "    lumi_weight    = %.4g\n"
            "    sample scale   = %.4g\n"
            "    total scale    = %.4g",
            sample_name, xsec, luminosity, n_gen,
            lumi_weight, sample.scale, total_scale,
        )

    branches = set()
    branches |= extract_branch_names(abcd_cfg.mass_variable)
    branches |= extract_branch_names(abcd_cfg.z_variable)
    branches |= extract_branch_names(region.selection)
    branches |= extract_branch_names(sample.weight_expr)
    if sample_cfg.selection:
        branches |= extract_branch_names(sample_cfg.selection)

    data = sample.load(branches, aliases=effective_aliases)

    mask = region.apply(data)
    if sample_cfg.selection:
        mask = mask & np.asarray(
            safe_evaluate(sample_cfg.selection, data), dtype=bool
        )

    mass_arr = np.asarray(safe_evaluate(abcd_cfg.mass_variable, data, mask=mask), dtype=float)
    z_arr    = np.asarray(safe_evaluate(abcd_cfg.z_variable,    data, mask=mask), dtype=float)

    total_scale = sample.scale * lumi_weight
    raw_w = safe_evaluate(sample.weight_expr, data, mask=mask)
    if np.ndim(raw_w) == 0 and float(raw_w) == 1.0 and total_scale == 1.0:
        weights = None
    else:
        w = np.asarray(raw_w, dtype=float)
        if np.ndim(w) == 0:
            w = np.full(int(np.sum(mask)), float(w))
        weights = w * total_scale

    n = len(mass_centers)
    na, na_err = np.zeros(n), np.zeros(n)
    nb, nc, nd = np.zeros(n), np.zeros(n), np.zeros(n)

    for i, mc in enumerate(mass_centers):
        hw, gap, sw = _compute_window(abcd_cfg, region.name, mc)
        c = _count_abcd(mass_arr, z_arr, mc, hw, gap, sw, abcd_cfg, weights)
        na[i], na_err[i] = c["A"]
        nb[i]            = c["B"][0]
        nc[i]            = c["C"][0]
        nd[i]            = c["D"][0]

    logger.info("Loaded %s: %.1f events in signal region A (total)", sample_name, na.sum())
    return {"na": na, "na_err": na_err, "nb": nb, "nc": nc, "nd": nd}


def _abcd_from_counts(nb, nc, nd):
    """Vectorised ABCD estimate over mass bins. Returns (est_arr, err_arr)."""
    n = len(nb)
    est = np.zeros(n)
    err = np.zeros(n)
    for i in range(n):
        est[i], err[i] = estimate_background(nb[i], nc[i], nd[i])
    return est, err


def _integrate_region(counts_dict, mass_centers, step, abcd_cfg, region_name):
    """Deweight sliding-window counts to produce a proper event integral.

    Each window of half-width hw_i centred on mass_centers[i] contains the
    same event ~(2*hw_i / step) times across neighbouring windows.  The factor
    w_i = step / (2*hw_i) converts per-window counts back to a per-GeV density
    summed over step → total events in the scanned mass range.

    Returns
    -------
    na_total, na_err_total, bkg_total, bkg_err_total : float
    """
    hw_arr = np.array(
        [_compute_window(abcd_cfg, region_name, mc)[0] for mc in mass_centers]
    )
    dw = step / (2.0 * hw_arr)

    na      = counts_dict["na"]
    na_err  = counts_dict["na_err"]
    bkg_arr, bkg_err_arr = _abcd_from_counts(
        counts_dict["nb"], counts_dict["nc"], counts_dict["nd"]
    )

    na_total      = float(np.sum(na * dw))
    na_err_total  = float(np.sqrt(np.sum((na_err * dw) ** 2)))
    bkg_total     = float(np.sum(bkg_arr * dw))
    bkg_err_total = float(np.sqrt(np.sum((bkg_err_arr * dw) ** 2)))
    return na_total, na_err_total, bkg_total, bkg_err_total


def plot_abcd_summary(plot_cfg, config, samples_map, output_dir, output_format,
                      counts_cache=None):
    """Summary bar chart with one bin per region.

    For each region in plot_cfg.regions, integrates the sliding-window ABCD
    estimate and observed data N_A (correcting for window overlap) to produce
    a single integrated count, then plots all regions side-by-side.

    Panels
    ------
    Top   : MC truth (navy bar), ABCD estimate (red hatched bar), data (black points)
    Bottom: ABCD / data  (red circles)  and  ABCD / MC truth  (navy squares)
    """
    abcd_cfg = plot_cfg.abcd
    step     = abcd_cfg.mass_scan_step

    mass_centers = np.arange(
        abcd_cfg.mass_scan_min,
        abcd_cfg.mass_scan_max + step * 0.5,
        step,
    )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_names = [s for s in plot_cfg.samples if samples_map[s].sample_type == "data"]
    mc_names   = [s for s in plot_cfg.samples if samples_map[s].sample_type != "data"]

    region_labels = []
    obs_vals,  obs_errs  = [], []
    bkg_vals,  bkg_errs  = [], []
    mc_vals,   mc_errs   = [], []

    for region_name in plot_cfg.regions:
        region_cfg_obj = next((r for r in config.regions if r.name == region_name), None)
        if region_cfg_obj is None:
            logger.warning("Region '%s' not found in summary, skipping.", region_name)
            continue

        region = Region(region_cfg_obj)
        region_labels.append(region_cfg_obj.label or region_name)

        if counts_cache and region_name in counts_cache:
            counts = counts_cache[region_name]
        else:
            counts = {}
            for name in mc_names + data_names:
                cfg_s = samples_map[name]
                ea    = {**config.aliases, **cfg_s.aliases}
                counts[name] = _compute_sample_counts(
                    name, cfg_s, mass_centers, abcd_cfg, region, ea, config.luminosity
                )

        # MC truth integral
        mc_na      = np.zeros(len(mass_centers))
        mc_na_err2 = np.zeros(len(mass_centers))
        for name in mc_names:
            mc_na      += counts[name]["na"]
            mc_na_err2 += counts[name]["na_err"] ** 2
        mc_na_dict = {"na": mc_na, "na_err": np.sqrt(mc_na_err2),
                      "nb": np.zeros_like(mc_na), "nc": np.zeros_like(mc_na),
                      "nd": np.zeros_like(mc_na)}
        mc_tot, mc_tot_err, _, _ = _integrate_region(
            mc_na_dict, mass_centers, step, abcd_cfg, region_name
        )
        mc_vals.append(mc_tot)
        mc_errs.append(mc_tot_err)

        # Data observed and ABCD estimate
        obs_tot = obs_err2 = 0.0
        bkg_tot = bkg_err2 = 0.0
        for name in data_names:
            na_t, na_e, bg_t, bg_e = _integrate_region(
                counts[name], mass_centers, step, abcd_cfg, region_name
            )
            obs_tot  += na_t
            obs_err2 += na_e ** 2
            bkg_tot  += bg_t
            bkg_err2 += bg_e ** 2
        obs_vals.append(obs_tot)
        obs_errs.append(sqrt(obs_err2))
        bkg_vals.append(bkg_tot)
        bkg_errs.append(sqrt(bkg_err2))

    if not region_labels:
        logger.warning("No regions found for ABCD summary plot '%s'.", plot_cfg.name)
        return

    n_regions = len(region_labels)
    x = np.arange(n_regions)
    bw = 0.3  # bar half-width

    # --- build figure -------------------------------------------------------
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(max(6, 2 * n_regions + 2), 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)
    add_hps_label(ax, lumi=config.luminosity)

    obs_vals  = np.array(obs_vals)
    obs_errs  = np.array(obs_errs)
    bkg_vals  = np.array(bkg_vals)
    bkg_errs  = np.array(bkg_errs)
    mc_vals   = np.array(mc_vals)
    mc_errs   = np.array(mc_errs)

    has_mc   = len(mc_names)   > 0
    has_data = len(data_names) > 0

    if has_mc:
        ax.bar(x - bw / 2, mc_vals, width=bw, color="navy", alpha=0.7,
               label="MC total (truth)", zorder=2)
        ax.errorbar(x - bw / 2, mc_vals, yerr=mc_errs,
                    fmt="none", color="navy", capsize=4, zorder=3)

    ax.bar(x + bw / 2, bkg_vals, width=bw, color="red", alpha=0.0,
           edgecolor="red", hatch="//", label="ABCD estimate", zorder=2)
    ax.errorbar(x + bw / 2, bkg_vals, yerr=bkg_errs,
                fmt="none", color="red", capsize=4, zorder=3)

    if has_data:
        ax.errorbar(x, obs_vals, yerr=obs_errs,
                    fmt="o", color="black", markersize=6,
                    label="Data (obs.)", zorder=4)

    ax.set_ylabel("Events (integrated)")
    if abcd_cfg.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * (10 if abcd_cfg.log_y else 1.5))
    ax.legend(fontsize=10)

    # --- ratio panel --------------------------------------------------------
    ax_r.axhline(1.0, color="gray", lw=1, ls="--")

    if has_data:
        valid_obs = obs_vals > 0
        ratio_obs     = np.where(valid_obs, bkg_vals / obs_vals, np.nan)
        ratio_obs_err = np.where(valid_obs, bkg_errs / obs_vals, np.nan)
        ax_r.errorbar(x, ratio_obs, yerr=ratio_obs_err,
                      fmt="o", color="red", markersize=5,
                      label="ABCD / obs.")

    if has_mc:
        valid_mc = mc_vals > 0
        ratio_mc     = np.where(valid_mc, bkg_vals / mc_vals, np.nan)
        ratio_mc_err = np.where(valid_mc, bkg_errs / mc_vals, np.nan)
        ax_r.errorbar(x, ratio_mc, yerr=ratio_mc_err,
                      fmt="s", color="navy", markersize=5,
                      label="ABCD / MC truth")

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(region_labels, rotation=15, ha="right")
    ax_r.set_ylabel("ABCD / Obs.")
    ax_r.set_ylim(0, 3)
    ax_r.legend(fontsize=9)

    fname = f"{plot_cfg.name}_summary.{output_format}"
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outdir / fname)


def plot_abcd(plot_cfg, region_cfg, config, samples_map, output_dir, output_format):
    """Run ABCD background estimation for one region.

    MC samples (sample_type != "data") are summed into a total background
    prediction.  Each process's truth N_A count is shown individually, plus the
    sum.  ABCD is run on the MC sum (closure test) and on data (data-driven
    estimate).  A ratio panel shows ABCD / observed for both.

    Parameters
    ----------
    plot_cfg : PlotConfig
    region_cfg : RegionConfig
    config : Config
    samples_map : dict[str, SampleConfig]
    output_dir : str
    output_format : str
    """
    abcd_cfg    = plot_cfg.abcd
    region      = Region(region_cfg)
    region_name = region_cfg.name

    mass_centers = np.arange(
        abcd_cfg.mass_scan_min,
        abcd_cfg.mass_scan_max + abcd_cfg.mass_scan_step * 0.5,
        abcd_cfg.mass_scan_step,
    )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- classify samples ---------------------------------------------------
    data_names = [s for s in plot_cfg.samples if samples_map[s].sample_type == "data"]
    mc_names   = [s for s in plot_cfg.samples if samples_map[s].sample_type != "data"]

    # --- load counts per sample ---------------------------------------------
    counts = {}
    for name in mc_names + data_names:
        cfg_s = samples_map[name]
        ea    = {**config.aliases, **cfg_s.aliases}
        counts[name] = _compute_sample_counts(
            name, cfg_s, mass_centers, abcd_cfg, region, ea, config.luminosity
        )

    # --- sum MC truth (region A only; ABCD is not run on MC) ----------------
    n_bins = len(mass_centers)
    mc_na      = np.zeros(n_bins)
    mc_na_err2 = np.zeros(n_bins)
    for name in mc_names:
        mc_na      += counts[name]["na"]
        mc_na_err2 += counts[name]["na_err"] ** 2
    mc_na_err = np.sqrt(mc_na_err2)

    # --- build figure -------------------------------------------------------
    has_data = len(data_names) > 0
    has_mc   = len(mc_names)   > 0

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)
    add_hps_label(ax, lumi=config.luminosity, extra_lines=[region_cfg.label])

    # ---- main panel --------------------------------------------------------

    # Individual MC truth (faded markers+lines) — only when show_mc_components is set
    if abcd_cfg.show_mc_components:
        for name in mc_names:
            cfg_s = samples_map[name]
            na    = counts[name]["na"]
            ea    = counts[name]["na_err"]
            ax.errorbar(
                mass_centers, na, yerr=ea,
                fmt="o-", color=cfg_s.color, label=cfg_s.label,
                markersize=3, lw=1, alpha=0.7,
            )

    # MC total truth (thick line + band)
    if has_mc:
        ax.step(mass_centers, mc_na, where="mid",
                color="navy", lw=2, label="MC total (truth)")
        ax.fill_between(
            mass_centers, mc_na - mc_na_err, mc_na + mc_na_err,
            step="mid", color="navy", alpha=0.15,
        )

    # Data observed N_A and ABCD estimate
    for name in data_names:
        cfg_s = samples_map[name]
        na    = counts[name]["na"]
        ea    = counts[name]["na_err"]
        bkg, bkg_err = _abcd_from_counts(
            counts[name]["nb"], counts[name]["nc"], counts[name]["nd"]
        )

        ax.errorbar(
            mass_centers, na, yerr=ea,
            fmt="o", color=cfg_s.color, label=f"{cfg_s.label} (obs.)",
            markersize=4, zorder=5,
        )
        ax.step(mass_centers, bkg, where="mid",
                color="red", lw=2, ls="--", label=f"ABCD ({cfg_s.label})")
        ax.fill_between(
            mass_centers, bkg - bkg_err, bkg + bkg_err,
            step="mid", color="red", alpha=0.10,
        )

    ax.set_ylabel("Events")
    if abcd_cfg.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    # Extend top of y-axis to leave room for the HPS label and legend
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * (10 if abcd_cfg.log_y else 1.5))
    ax.legend(fontsize=9, loc="upper right")

    # ---- ratio panel: ABCD(data) / data N_A --------------------------------
    ax_r.axhline(1.0, color="gray", lw=1)

    for name in data_names:
        cfg_s = samples_map[name]
        na    = counts[name]["na"]
        bkg, bkg_err = _abcd_from_counts(
            counts[name]["nb"], counts[name]["nc"], counts[name]["nd"]
        )

        # ABCD(data) / data N_A
        valid     = na > 0
        ratio     = np.where(valid, bkg     / na, np.nan)
        ratio_err = np.where(valid, bkg_err / na, np.nan)
        ax_r.errorbar(
            mass_centers, ratio, yerr=ratio_err,
            fmt="o", color="red", markersize=3, lw=1,
            label=f"ABCD({cfg_s.label}) / obs.",
        )

        # ABCD(data) / MC total truth
        if has_mc:
            valid_mc     = mc_na > 0
            ratio_mc     = np.where(valid_mc, bkg     / mc_na, np.nan)
            ratio_mc_err = np.where(valid_mc, bkg_err / mc_na, np.nan)
            ax_r.errorbar(
                mass_centers, ratio_mc, yerr=ratio_mc_err,
                fmt="s", color="navy", markersize=3, lw=1,
                label=f"ABCD({cfg_s.label}) / MC truth",
            )

    ax_r.set_xlabel(abcd_cfg.x_label)
    ax_r.set_ylabel("ABCD / Obs.")
    ax_r.set_ylim(0, 3)
    ax_r.legend(fontsize=9, loc="upper right")

    # ---- save --------------------------------------------------------------
    fname = f"{plot_cfg.name}_{region_name}.{output_format}"
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outdir / fname)

    return counts
