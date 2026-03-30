"""ABCD background estimation method.

Scans over invariant mass bins and estimates background using:
    N_A(bkg) = N_B * N_C / N_D

Regions (defined per mass center):
    A: signal mass window  AND signal Z region    (observed signal-region count)
    B: signal mass window  AND control Z region
    C: sideband mass       AND signal Z region
    D: sideband mass       AND control Z region
"""

import copy
import json
import logging
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import add_hps_label
from ..region import Region
from ..sample import Sample, compute_luminosity
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


def _load_sample_raw(sample_name, sample_cfg, abcd_cfg, all_branches,
                     effective_aliases, luminosity, ann_scorer=None):
    """Load a sample's data without applying any region mask.

    Parameters
    ----------
    all_branches : set
        Union of branch names needed by all regions in this plot (plus the
        mass/z variables and weight expression).  Passing the full union once
        ensures the cached data can be reused for every region.
    ann_scorer : tuple or None

    Returns
    -------
    dict with keys:
        'data'        — raw event arrays keyed by branch name
        'total_scale' — combined lumi × sample scale factor
        'weight_expr' — weight expression string for this sample
    """
    from ..ann_classifier import (
        ANN_VTX_ALIASES, ANN_DIRECT_BRANCHES,
        build_ann_matrix, predict_scores,
    )

    sample = Sample(sample_cfg)

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

    branches = set(all_branches)
    branches |= extract_branch_names(abcd_cfg.mass_variable)
    branches |= extract_branch_names(abcd_cfg.z_variable)
    branches |= extract_branch_names(sample.weight_expr)
    if sample_cfg.selection:
        branches |= extract_branch_names(sample_cfg.selection)

    ann_score_var = None
    load_aliases = effective_aliases
    if ann_scorer is not None:
        ann_model, ann_mean, ann_scale, score_var = ann_scorer
        if score_var in branches:
            branches.discard(score_var)
            branches.update(ANN_DIRECT_BRANCHES)
            branches.update(ANN_VTX_ALIASES.keys())
            load_aliases = {**effective_aliases, **ANN_VTX_ALIASES}
            ann_score_var = score_var
            logger.info("ANN scoring enabled for sample '%s' (variable: %s)", sample_name, score_var)

    data = sample.load(branches, aliases=load_aliases)

    if ann_score_var is not None:
        X = build_ann_matrix(data)
        data[ann_score_var] = predict_scores(ann_model, ann_mean, ann_scale, X)
        logger.info("  ANN scores computed: mean=%.4f", float(data[ann_score_var].mean()))

    return {
        "data":        data,
        "total_scale": sample.scale * lumi_weight,
        "weight_expr": sample.weight_expr,
    }


def _count_from_raw(raw, region, sample_cfg, mass_centers, abcd_cfg):
    """Apply a region mask to pre-loaded data and return ABCD counts.

    Parameters
    ----------
    raw : dict
        Output of ``_load_sample_raw``.

    Returns
    -------
    dict with keys: na, na_err, nb, nc, nd — numpy arrays
    """
    data        = raw["data"]
    total_scale = raw["total_scale"]
    weight_expr = raw["weight_expr"]

    mask = region.apply(data)
    if sample_cfg.selection:
        mask = mask & np.asarray(safe_evaluate(sample_cfg.selection, data), dtype=bool)

    mass_arr = np.asarray(safe_evaluate(abcd_cfg.mass_variable, data, mask=mask), dtype=float)
    z_arr    = np.asarray(safe_evaluate(abcd_cfg.z_variable,    data, mask=mask), dtype=float)

    raw_w = safe_evaluate(weight_expr, data, mask=mask)
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

    logger.info("Counted %s / %s: %.1f events in signal region A",
                region.name, region.name, na.sum())
    return {"na": na, "na_err": na_err, "nb": nb, "nc": nc, "nd": nd}


def _compute_sample_counts(sample_name, sample_cfg, mass_centers, abcd_cfg,
                           region, effective_aliases, luminosity, ann_scorer=None):
    """Load a sample and return per-mass-bin ABCD region counts.

    Thin wrapper around ``_load_sample_raw`` + ``_count_from_raw`` for
    call sites that don't use the data cache (summary / lumi-projection
    fallback paths).
    """
    raw = _load_sample_raw(
        sample_name, sample_cfg, abcd_cfg,
        extract_branch_names(region.selection),
        effective_aliases, luminosity, ann_scorer,
    )
    return _count_from_raw(raw, region, sample_cfg, mass_centers, abcd_cfg)


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
                      counts_cache=None, ann_scorer=None):
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
                    name, cfg_s, mass_centers, abcd_cfg, region, ea, config.luminosity,
                    ann_scorer=ann_scorer,
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


def plot_abcd(plot_cfg, region_cfg, config, samples_map, output_dir, output_format,
             ann_scorer=None, data_cache=None, all_region_branches=None):
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
    # When data_cache is provided (shared across region calls) each sample is
    # loaded from disk only once; subsequent regions reuse the cached arrays
    # and just re-apply the region mask via _count_from_raw.
    counts = {}
    for name in mc_names + data_names:
        cfg_s = samples_map[name]
        ea    = {**config.aliases, **cfg_s.aliases}
        if data_cache is not None:
            if name not in data_cache:
                data_cache[name] = _load_sample_raw(
                    name, cfg_s, abcd_cfg,
                    all_region_branches or set(),
                    ea, config.luminosity, ann_scorer,
                )
            counts[name] = _count_from_raw(
                data_cache[name], region, cfg_s, mass_centers, abcd_cfg,
            )
        else:
            counts[name] = _compute_sample_counts(
                name, cfg_s, mass_centers, abcd_cfg, region, ea, config.luminosity,
                ann_scorer=ann_scorer,
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


def plot_abcd_lumi_projections(plot_cfg, config, samples_map, output_dir, output_format,
                               counts_cache=None, ann_scorer=None):
    """Plot ABCD background estimates scaled to projected luminosities.

    For each LumiProjectionConfig in plot_cfg.abcd.lumi_projections:
    - ``directory`` mode: compute luminosity from that dataset dynamically, scale
      the reference ABCD estimate by L_proj/L_ref, and overlay observed N_A from
      that dataset for comparison.
    - ``scale_factor`` mode: scale the reference ABCD estimate by the given
      factor (projection only — no observed data).

    One plot is produced per region.
    """
    abcd_cfg = plot_cfg.abcd
    if not abcd_cfg.lumi_projections:
        return

    step = abcd_cfg.mass_scan_step
    mass_centers = np.arange(
        abcd_cfg.mass_scan_min,
        abcd_cfg.mass_scan_max + step * 0.5,
        step,
    )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_names = [s for s in plot_cfg.samples if samples_map[s].sample_type == "data"]
    L_ref = config.luminosity

    for region_name in plot_cfg.regions:
        region_cfg_obj = next((r for r in config.regions if r.name == region_name), None)
        if region_cfg_obj is None:
            logger.warning("Region '%s' not found, skipping lumi projection.", region_name)
            continue
        region = Region(region_cfg_obj)

        # Get reference ABCD counts (use cache from plot_abcd when available)
        if counts_cache and region_name in counts_cache:
            ref_counts = counts_cache[region_name]
        else:
            ref_counts = {}
            for name in data_names:
                cfg_s = samples_map[name]
                ea = {**config.aliases, **cfg_s.aliases}
                ref_counts[name] = _compute_sample_counts(
                    name, cfg_s, mass_centers, abcd_cfg, region, ea, L_ref,
                    ann_scorer=ann_scorer,
                )

        # Sum reference ABCD estimate over all data samples
        ref_bkg = np.zeros(len(mass_centers))
        ref_bkg_err2 = np.zeros(len(mass_centers))
        for name in data_names:
            if name in ref_counts:
                b, be = _abcd_from_counts(
                    ref_counts[name]["nb"], ref_counts[name]["nc"], ref_counts[name]["nd"]
                )
                ref_bkg += b
                ref_bkg_err2 += be ** 2
        ref_bkg_err = np.sqrt(ref_bkg_err2)

        # --- build figure ---------------------------------------------------
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(10, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.05)
        add_hps_label(ax, lumi=L_ref, extra_lines=[region_cfg_obj.label])

        # Draw reference ABCD estimate
        ax.step(mass_centers, ref_bkg, where="mid",
                color="black", lw=1.5, ls=":", alpha=0.6,
                label=f"ABCD ref. ({L_ref:.3g} pb$^{{-1}}$)", zorder=2)
        ax.fill_between(
            mass_centers, ref_bkg - ref_bkg_err, ref_bkg + ref_bkg_err,
            step="mid", color="black", alpha=0.08, zorder=1,
        )

        has_ratio = False
        proj_ratio_data = []  # (mass_centers, ratio, ratio_err, color, label)

        for proj in abcd_cfg.lumi_projections:
            if proj.directory:
                lumi_file_to_use = proj.lumi_file or config.lumi_file
                if not lumi_file_to_use:
                    logger.warning(
                        "Lumi projection '%s' has directory but no lumi_file — skipping.", proj.label
                    )
                    continue
                L_val = compute_luminosity([proj.directory], lumi_file_to_use)
                scale = L_val / L_ref if L_ref > 0 else 0.0
                logger.info(
                    "Lumi projection '%s': L=%.4g pb⁻¹, scale=%.4g", proj.label, L_val, scale
                )

                # Load observed data from the projection directory
                proj_na = np.zeros(len(mass_centers))
                proj_na_err2 = np.zeros(len(mass_centers))
                load_ok = True
                for name in data_names:
                    orig_cfg = samples_map[name]
                    proj_cfg = copy.copy(orig_cfg)
                    proj_cfg.directory = proj.directory
                    proj_cfg.directories = [proj.directory]
                    proj_cfg.run_min = None
                    proj_cfg.run_max = None
                    ea = {**config.aliases, **proj_cfg.aliases}
                    try:
                        pc = _compute_sample_counts(
                            f"{name} ({proj.label})", proj_cfg,
                            mass_centers, abcd_cfg, region, ea, L_val,
                            ann_scorer=ann_scorer,
                        )
                        proj_na += pc["na"]
                        proj_na_err2 += pc["na_err"] ** 2
                    except Exception as exc:
                        logger.warning(
                            "Failed to load lumi projection '%s' dir '%s': %s",
                            proj.label, proj.directory, exc,
                        )
                        load_ok = False

                has_observed = load_ok
                proj_na_err = np.sqrt(proj_na_err2)

            else:
                scale = proj.scale_factor
                L_val = L_ref * scale
                logger.info(
                    "Lumi projection '%s': scale=%.4g (L≈%.4g pb⁻¹)", proj.label, scale, L_val
                )
                has_observed = False
                proj_na = proj_na_err = None

            # Scaled ABCD estimate
            scaled_bkg = ref_bkg * scale
            scaled_bkg_err = ref_bkg_err * scale

            ax.step(mass_centers, scaled_bkg, where="mid",
                    color=proj.color, lw=2, ls=proj.linestyle,
                    label=f"ABCD {proj.label}", zorder=3)
            ax.fill_between(
                mass_centers, scaled_bkg - scaled_bkg_err, scaled_bkg + scaled_bkg_err,
                step="mid", color=proj.color, alpha=0.10, zorder=2,
            )

            if has_observed:
                ax.errorbar(
                    mass_centers, proj_na, yerr=proj_na_err,
                    fmt="o", color=proj.color, markersize=4,
                    label=f"{proj.label} (obs.)", zorder=5,
                )
                valid = scaled_bkg > 0
                ratio = np.where(valid, proj_na / scaled_bkg, np.nan)
                ratio_err = np.where(valid, proj_na_err / scaled_bkg, np.nan)
                proj_ratio_data.append((mass_centers, ratio, ratio_err, proj.color, proj.label))
                has_ratio = True

        ax.set_ylabel("Events")
        if abcd_cfg.log_y:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.1)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * (10 if abcd_cfg.log_y else 1.5))
        ax.legend(fontsize=9, loc="upper right")

        # --- ratio panel ----------------------------------------------------
        ax_r.axhline(1.0, color="gray", lw=1, ls="--")
        if has_ratio:
            for mc, ratio, ratio_err, color, label in proj_ratio_data:
                ax_r.errorbar(
                    mc, ratio, yerr=ratio_err,
                    fmt="o", color=color, markersize=3, lw=1,
                    label=f"obs. / ABCD ({label})",
                )
            ax_r.legend(fontsize=9, loc="upper right")

        ax_r.set_xlabel(abcd_cfg.x_label)
        ax_r.set_ylabel("Obs. / ABCD")
        ax_r.set_ylim(0, 3)

        fname = f"{plot_cfg.name}_{region_name}_lumi_proj.{output_format}"
        fig.savefig(outdir / fname, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", outdir / fname)


def write_abcd_json(plot_cfg, config, samples_map, counts_cache, output_path):
    """Write per-mass-bin ABCD results to a JSON file.

    Structure
    ---------
    {
      "meta": { plot name, luminosity, z-axis settings, ... },
      "regions": {
        "<region>": [
          {
            "mass_GeV": ...,
            "resolution_GeV": ...,   # null when fixed (non-sigma) windows used
            "signal_window_GeV": ...,
            "sideband_gap_GeV": ...,
            "sideband_width_GeV": ...,
            "n_a_obs": ..., "n_a_obs_err": ...,
            "bkg_est": ..., "bkg_est_err": ...,
            "projections": { "<label>": {"bkg_est": ..., "bkg_est_err": ...} }
          }, ...
        ]
      }
    }
    """
    abcd_cfg = plot_cfg.abcd

    mass_centers = np.arange(
        abcd_cfg.mass_scan_min,
        abcd_cfg.mass_scan_max + abcd_cfg.mass_scan_step * 0.5,
        abcd_cfg.mass_scan_step,
    )

    data_names = [s for s in plot_cfg.samples if samples_map[s].sample_type == "data"]
    L_ref = config.luminosity

    # Resolve projection scale factors
    proj_scales = {}
    for proj in abcd_cfg.lumi_projections:
        if proj.scale_factor:
            proj_scales[proj.label] = proj.scale_factor
        elif proj.directory:
            lumi_file = proj.lumi_file or config.lumi_file
            if lumi_file:
                L_val = compute_luminosity([proj.directory], lumi_file)
                proj_scales[proj.label] = L_val / L_ref if L_ref > 0 else 0.0

    meta = {
        "plot_name": plot_cfg.name,
        "luminosity_pb": L_ref,
        "z_variable": abcd_cfg.z_variable,
        "z_signal": [abcd_cfg.z_signal_min, abcd_cfg.z_signal_max],
        "z_control": [abcd_cfg.z_control_min, abcd_cfg.z_control_max],
        "sr_sigmas": abcd_cfg.sr_sigmas,
        "gap_sigmas": abcd_cfg.gap_sigmas,
        "sb_sigmas": abcd_cfg.sb_sigmas,
    }

    regions_data = {}

    for region_name, counts in counts_cache.items():
        # Sum observed N_A and ABCD inputs across all data samples
        na      = np.zeros(len(mass_centers))
        na_err2 = np.zeros(len(mass_centers))
        nb      = np.zeros(len(mass_centers))
        nc      = np.zeros(len(mass_centers))
        nd      = np.zeros(len(mass_centers))

        for name in data_names:
            if name in counts:
                na      += counts[name]["na"]
                na_err2 += counts[name]["na_err"] ** 2
                nb      += counts[name]["nb"]
                nc      += counts[name]["nc"]
                nd      += counts[name]["nd"]

        na_err = np.sqrt(na_err2)
        bkg, bkg_err = _abcd_from_counts(nb, nc, nd)

        mass_points = []
        res_expr = abcd_cfg.mass_resolution.get(region_name)

        for i, mc in enumerate(mass_centers):
            hw, gap, sw = _compute_window(abcd_cfg, region_name, mc)

            if res_expr:
                sigma = float(safe_evaluate(res_expr, {"m": np.array([mc])}))
            else:
                sigma = None

            entry = {
                "mass_GeV":          round(float(mc), 6),
                "resolution_GeV":    round(float(sigma), 6) if sigma is not None else None,
                "signal_window_GeV": round(float(hw), 6),
                "sideband_gap_GeV":  round(float(gap), 6),
                "sideband_width_GeV": round(float(sw), 6),
                "n_a_obs":     round(float(na[i]), 4),
                "n_a_obs_err": round(float(na_err[i]), 4),
                "bkg_est":     round(float(bkg[i]), 4),
                "bkg_est_err": round(float(bkg_err[i]), 4),
            }

            if proj_scales:
                entry["projections"] = {
                    label: {
                        "bkg_est":     round(float(bkg[i] * scale), 4),
                        "bkg_est_err": round(float(bkg_err[i] * scale), 4),
                    }
                    for label, scale in proj_scales.items()
                }

            mass_points.append(entry)

        regions_data[region_name] = mass_points

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "regions": regions_data}, f, indent=2)
    logger.info("ABCD results JSON written: %s", out_path)
