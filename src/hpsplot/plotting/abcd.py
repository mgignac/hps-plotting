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
import re as _re
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import add_hps_label
from ..region import Region
from ..sample import Sample, compute_luminosity
from ..utils import safe_evaluate, extract_branch_names
from .. import simp_scaling
from ..signal_scaling import sarah_prompt_yield

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
    # For auto-generated aux bin regions (name contains __auxvar__) fall back to
    # the base region's mass resolution entry.
    if not res_expr:
        m = _re.match(r'^(.+)__auxvar__.*$', region_name)
        if m:
            res_expr = abcd_cfg.mass_resolution.get(m.group(1))
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
        "A":         _sum(mass_sig & z_sig),
        "B":         _sum(mass_sig & z_ctrl),
        "C":         _sum(mass_sb  & z_sig),
        "D":         _sum(mass_sb  & z_ctrl),
        "mass_only": _sum(mass_sig),          # mass window only, no z cut — used for Eq. 4
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
    n_mass_only = np.zeros(n)  # mass window only, no z cut — Eq. 4 numerator for data

    for i, mc in enumerate(mass_centers):
        hw, gap, sw = _compute_window(abcd_cfg, region.name, mc)
        c = _count_abcd(mass_arr, z_arr, mc, hw, gap, sw, abcd_cfg, weights)
        na[i], na_err[i] = c["A"]
        nb[i]            = c["B"][0]
        nc[i]            = c["C"][0]
        nd[i]            = c["D"][0]
        n_mass_only[i]   = c["mass_only"][0]

        if nd[i] > 0:
            est = nb[i] * nc[i] / nd[i]
        else:
            est = 0.0
        logger.debug(
            "  [%s | m=%.1f MeV]  "
            "A (sig): mass [%.4f, %.4f], z [%.2f, %.2f]  "
            "B (z-ctrl): mass [%.4f, %.4f], z [%.2f, %.2f]  "
            "C (sb-sig): mass [%.4f,%.4f]|[%.4f,%.4f], z [%.2f, %.2f]  "
            "D (sb-ctrl): same mass as C, z [%.2f, %.2f]  |  "
            "N_A=%g  N_B=%g  N_C=%g  N_D=%g  "
            "est = N_B*N_C/N_D = %g",
            region.name, mc * 1000,
            # A
            mc - hw, mc + hw,
            abcd_cfg.z_signal_min, abcd_cfg.z_signal_max,
            # B
            mc - hw, mc + hw,
            abcd_cfg.z_control_min, abcd_cfg.z_control_max,
            # C
            mc - hw - gap - sw, mc - hw - gap,
            mc + hw + gap,       mc + hw + gap + sw,
            abcd_cfg.z_signal_min, abcd_cfg.z_signal_max,
            # D
            abcd_cfg.z_control_min, abcd_cfg.z_control_max,
            # counts
            na[i], nb[i], nc[i], nd[i], est,
        )

    logger.info("Counted %s: N_A=%.1f  (total over %d mass bins)",
                region.name, na.sum(), n)
    return {"na": na, "na_err": na_err, "nb": nb, "nc": nc, "nd": nd,
            "n_mass_only": n_mass_only}


_ALPHA_EM = 1.0 / 137.036


def _eval_rad_frac(rad_frac, mass_center):
    """Return the radiative fraction at *mass_center* [GeV].

    *rad_frac* may be a plain float or a string expression in ``m`` (mass in GeV).
    """
    if isinstance(rad_frac, str):
        return float(safe_evaluate(rad_frac, {"m": np.array([mass_center])}))
    return float(rad_frac)


def _count_signal_scan_from_raw(raw, region, sample_cfg, mass_centers, abcd_cfg,
                                 rad_frac, n_data_mass_only, norm_region=None,
                                 data_mass_arr_norm=None):
    """Compute per-eps² signal yields in region A with Eq. 4 scaling.

    Handles both ``weight_scan`` entries (explicit weight expressions) and
    ``eps2_scan`` (numerical lifetime reweighting, cτ derived from ε² and ap_mass).

    Returns list of dicts, one per (eps2, label):
        label, ap_mass, eps2, n_mc_a, n_mc_total, nsig, norm_region, window_scan
    """
    has_ws = bool(sample_cfg.weight_scan)
    has_e2 = sample_cfg.eps2_scan is not None
    if not has_ws and not has_e2:
        return []

    data        = raw["data"]
    total_scale = raw["total_scale"]
    ap_mass     = sample_cfg.ap_mass
    eff_norm    = norm_region if norm_region is not None else region

    # --- masks and kinematic arrays -----------------------------------------
    abcd_mask = region.apply(data)
    if sample_cfg.selection:
        abcd_mask = abcd_mask & np.asarray(safe_evaluate(sample_cfg.selection, data), dtype=bool)

    mass_arr_abcd = np.asarray(safe_evaluate(abcd_cfg.mass_variable, data, mask=abcd_mask), dtype=float)
    z_arr_abcd    = np.asarray(safe_evaluate(abcd_cfg.z_variable,    data, mask=abcd_mask), dtype=float)

    norm_mask = eff_norm.apply(data)
    if sample_cfg.selection:
        norm_mask = norm_mask & np.asarray(safe_evaluate(sample_cfg.selection, data), dtype=bool)

    mass_arr_norm = np.asarray(safe_evaluate(abcd_cfg.mass_variable, data, mask=norm_mask), dtype=float)
    z_arr_norm    = np.asarray(safe_evaluate(abcd_cfg.z_variable,    data, mask=norm_mask), dtype=float)

    n = len(mass_centers)

    # --- build scan items: (eps2, label, weights_abcd, weights_norm, verbose) -
    # verbose=True  → emit per-mass-center INFO log (weight_scan, small N of entries)
    # verbose=False → emit only the per-eps² summary line (eps2_scan, can be O(10s) of entries)
    scan_items = []

    for entry in sample_cfg.weight_scan:
        if entry.epsilon_sq is None:
            logger.debug("WeightScanEntry '%s' has no epsilon_sq — skipping Eq. 4.", entry.label)
            continue
        raw_w = safe_evaluate(entry.weight, data, mask=abcd_mask)
        w = (np.full(int(np.sum(abcd_mask)), float(raw_w))
             if np.ndim(raw_w) == 0 else np.asarray(raw_w, dtype=float))
        wa = np.where(np.isfinite(w * total_scale) & (w * total_scale >= 0.0),
                      w * total_scale, 0.0)
        raw_w_n = safe_evaluate(entry.weight, data, mask=norm_mask)
        w_n = (np.full(int(np.sum(norm_mask)), float(raw_w_n))
               if np.ndim(raw_w_n) == 0 else np.asarray(raw_w_n, dtype=float))
        wn = np.where(np.isfinite(w_n * total_scale) & (w_n * total_scale >= 0.0),
                      w_n * total_scale, 0.0)
        scan_items.append((entry.epsilon_sq, entry.label, wa, wn, True))

    if has_e2:
        es = sample_cfg.eps2_scan
        if ap_mass is None:
            logger.warning("eps2_scan on '%s' requires ap_mass to be set — skipping.", sample_cfg.name)
        elif sample_cfg.signal_type == "simp":
            # ---- SIMP path: dark vector lifetime reweighting ----------------
            if sample_cfg.simp is None:
                logger.warning(
                    "signal_type='simp' on '%s' but no simp: config block — skipping SIMP scan.",
                    sample_cfg.name,
                )
            else:
                sc          = sample_cfg.simp
                ap_mass_mev = ap_mass * 1000.0
                m_pi_D, m_V_D, f_pi_D = simp_scaling.dark_masses(
                    ap_mass_mev, sc.vd_mass_ratio, sc.pid_mass_ratio)
                rho_frac, phi_frac = simp_scaling.dark_branching_fractions(
                    sc.alpha_d, m_pi_D, m_V_D, f_pi_D, ap_mass_mev)

                # Load dark-vector truth arrays (z and βγ of V_D, not A')
                z_vd_a  = np.asarray(safe_evaluate(es.z_branch,         data, mask=abcd_mask), dtype=float)
                bg_vd_a = np.asarray(safe_evaluate(es.betagamma_branch, data, mask=abcd_mask), dtype=float)
                bw_a    = np.asarray(safe_evaluate(es.base_weight,      data, mask=abcd_mask), dtype=float) * total_scale

                z_vd_n  = np.asarray(safe_evaluate(es.z_branch,         data, mask=norm_mask), dtype=float)
                bg_vd_n = np.asarray(safe_evaluate(es.betagamma_branch, data, mask=norm_mask), dtype=float)
                bw_n    = np.asarray(safe_evaluate(es.base_weight,      data, mask=norm_mask), dtype=float) * total_scale

                logger.debug(
                    "SIMP dark sector [m_A'=%.1f MeV, alpha_d=%.3g]: "
                    "m_pi_D=%.2f MeV  m_V_D=%.2f MeV  f_pi_D=%.4g MeV  "
                    "rho_frac=%.4f  phi_frac=%.4f  BR_vis=%.4f",
                    ap_mass_mev, sc.alpha_d,
                    m_pi_D, m_V_D, f_pi_D, rho_frac, phi_frac, rho_frac + phi_frac,
                )

                for eps2 in es.eps2_values:
                    ctau_rho, ctau_phi = simp_scaling.dark_vector_ctau(
                        sc.alpha_d, m_pi_D, m_V_D, f_pi_D, ap_mass_mev, eps2)
                    wa = simp_scaling.simp_event_weights(
                        z_vd_a, bg_vd_a, bw_a, es.gen_length_mm,
                        ctau_rho, ctau_phi, rho_frac, phi_frac, es.target_z_mm)
                    wn = simp_scaling.simp_event_weights(
                        z_vd_n, bg_vd_n, bw_n, es.gen_length_mm,
                        ctau_rho, ctau_phi, rho_frac, phi_frac, es.target_z_mm)
                    logger.info(
                        "  SIMP eps2=%.3g: ctau_rho=%.3g mm  ctau_phi=%.3g mm  "
                        "n_events_abcd=%d  sum_weights_abcd=%.4g",
                        eps2, ctau_rho, ctau_phi, len(wa), float(np.sum(wa)),
                    )
                    label = f"{sample_cfg.name}_simp_eps2_{eps2:.3e}"
                    scan_items.append((eps2, label, wa, wn, False))
        else:
            # ---- standard A' eps2_scan (unchanged) --------------------------
            ap_mass_mev = ap_mass * 1000.0
            z_a  = np.asarray(safe_evaluate(es.z_branch,         data, mask=abcd_mask), dtype=float)
            bg_a = np.asarray(safe_evaluate(es.betagamma_branch, data, mask=abcd_mask), dtype=float)
            bw_a = np.asarray(safe_evaluate(es.base_weight,      data, mask=abcd_mask), dtype=float) * total_scale

            z_n  = np.asarray(safe_evaluate(es.z_branch,         data, mask=norm_mask), dtype=float)
            bg_n = np.asarray(safe_evaluate(es.betagamma_branch, data, mask=norm_mask), dtype=float)
            bw_n = np.asarray(safe_evaluate(es.base_weight,      data, mask=norm_mask), dtype=float) * total_scale

            for eps2 in es.eps2_values:
                ctau = 8.109e-8 / (eps2 * ap_mass_mev)

                sh_a = z_a - es.target_z_mm
                L_a  = bg_a * ctau
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    lt_a = np.where((sh_a >= 0) & (L_a > 0), np.exp(-sh_a / L_a) / L_a, 0.0)
                wa = np.where(np.isfinite(bw_a * lt_a) & (bw_a * lt_a >= 0),
                              bw_a * lt_a * es.gen_length_mm, 0.0)

                sh_n = z_n - es.target_z_mm
                L_n  = bg_n * ctau
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    lt_n = np.where((sh_n >= 0) & (L_n > 0), np.exp(-sh_n / L_n) / L_n, 0.0)
                wn = np.where(np.isfinite(bw_n * lt_n) & (bw_n * lt_n >= 0),
                              bw_n * lt_n * es.gen_length_mm, 0.0)

                label = f"{sample_cfg.name}_eps2_{eps2:.3e}"
                scan_items.append((eps2, label, wa, wn, False))

    # --- unified counting loop over all scan items --------------------------
    norm_name = eff_norm.config.name

    # Log normalisation setup once per sample so the user can see exactly what
    # region selection and mass window are applied to both data and signal MC.
    _snhw_fixed = abcd_cfg.signal_norm_hw  # None → uses per-bin ABCD hw
    _snhw_desc  = (f"signal_norm_hw = ±{_snhw_fixed*1000:.2f} MeV (fixed)"
                   if _snhw_fixed is not None
                   else "signal_norm_hw = ±hw(m)  (follows ABCD window)")
    logger.info(
        "Signal norm setup for '%s':\n"
        "  Norm region   : '%s'  selection = %s\n"
        "  ABCD region   : '%s'  selection = %s\n"
        "  Mass window   : %s\n"
        "  N_data_mass_only per bin (data passing norm-region selection AND mass window):\n"
        "    %s",
        sample_cfg.name,
        norm_name, eff_norm.config.selection,
        region.config.name, region.config.selection,
        _snhw_desc,
        "  ".join(
            f"m={mc*1000:.0f} MeV → N_data={n_data_mass_only[i]:.0f}"
            for i, mc in enumerate(mass_centers)
        ),
    )

    results   = []

    for eps2_val, label, weights_abcd, weights_norm, verbose in scan_items:
        n_mc_total     = float(np.sum(weights_norm))
        n_mc_a         = np.zeros(n)
        n_mc_mass_only = np.zeros(n)
        nsig           = np.zeros(n)

        for i, mc in enumerate(mass_centers):
            hw, gap, sw = _compute_window(abcd_cfg, region.name, mc)
            snhw = abcd_cfg.signal_norm_hw if abcd_cfg.signal_norm_hw is not None else hw

            c_abcd = _count_abcd(mass_arr_abcd, z_arr_abcd, mc, hw, gap, sw, abcd_cfg, weights_abcd)
            n_mc_a[i] = c_abcd["A"][0]

            m_sig_n = (mass_arr_norm >= mc - snhw) & (mass_arr_norm <= mc + snhw)
            n_mc_mass_only[i] = float(np.sum(weights_norm[m_sig_n]))

            if n_mc_mass_only[i] > 0 and n_data_mass_only[i] > 0:
                delta_m      = 2.0 * snhw
                mass_hyp     = ap_mass if ap_mass is not None else mc
                rad_frac_val = _eval_rad_frac(rad_frac, mc)
                n_expected   = sarah_prompt_yield(mass_hyp, eps2_val, n_data_mass_only[i], delta_m, rad_frac_val)
                scale        = n_expected / n_mc_mass_only[i]
                nsig[i]      = n_mc_a[i] * scale
                if verbose:
                    eps_z = n_mc_a[i] / n_mc_mass_only[i]
                    logger.info(
                        "  [%s | m=%.0f MeV | eps2=%.3g]\n"
                        "    Mass window   : [%.4f, %.4f] GeV  (hw=%.4f, delta_m=%.4f GeV)\n"
                        "    z cut         : min_y0 in [%.3f, %.3g]\n"
                        "    N_data_window : %.4g  (data in mass window, no z cut)\n"
                        "    N_mc_A        : %.4g  (MC in mass window AND z-signal cut)\n"
                        "    N_mc_window   : %.4g  (MC in mass window, no z cut)\n"
                        "    N_mc_total    : %.4g  (MC in full norm region, no cuts)\n"
                        "    eps_z         : %.4g  (N_mc_A / N_mc_window — z-cut efficiency)\n"
                        "    f_rad         : %.4g\n"
                        "    n_expected    : sarah_prompt_yield(...) = %.4g\n"
                        "    scale         : n_expected / N_mc_window = %.4g\n"
                        "    nsig          : N_mc_A * scale = %.4g",
                        region.config.name, mc * 1000, eps2_val,
                        mc - hw, mc + hw, hw, delta_m,
                        abcd_cfg.z_signal_min, abcd_cfg.z_signal_max,
                        n_data_mass_only[i], n_mc_a[i], n_mc_mass_only[i], n_mc_total,
                        eps_z, rad_frac_val,
                        n_expected, scale, nsig[i],
                    )
            elif verbose:
                logger.info(
                    "  [%s | m=%.0f MeV | eps2=%.3g]  SKIPPED: "
                    "N_mc_window=%.4g  N_data_window=%.4g",
                    region.config.name, mc * 1000, eps2_val,
                    n_mc_mass_only[i], n_data_mass_only[i],
                )

        # Report the delta_m actually used. When signal_norm_hw is fixed it is
        # constant; otherwise it is the ABCD hw at the sample's ap_mass (or the
        # first mass centre as a representative value).
        if abcd_cfg.signal_norm_hw is not None:
            _rep_dm = 2.0 * abcd_cfg.signal_norm_hw
        else:
            _rep_mc = ap_mass if ap_mass is not None else mass_centers[0]
            _rep_dm = 2.0 * _compute_window(abcd_cfg, region.name, _rep_mc)[0]
        logger.info(
            "Signal scan '%s' (eps2=%.3g) region '%s' [norm: '%s', delta_m=%.2f MeV]: "
            "N_mc_total=%.3g  N_sig(summed)=%.3g",
            label, eps2_val, region.config.name, norm_name, _rep_dm * 1000,
            n_mc_total, float(np.sum(nsig)),
        )

        ws_results = []
        if abcd_cfg.window_scan and data_mass_arr_norm is not None:
            z_sig_mask = (z_arr_abcd >= abcd_cfg.z_signal_min) & (z_arr_abcd <= abcd_cfg.z_signal_max)
            for hw_ws in abcd_cfg.window_scan:
                nsig_ws_arr = np.zeros(n)
                for i, mc in enumerate(mass_centers):
                    mass_hyp   = ap_mass if ap_mass is not None else mc
                    delta_m_ws = 2.0 * hw_ws
                    n_data_ws  = float(np.sum(
                        (data_mass_arr_norm >= mc - hw_ws) & (data_mass_arr_norm <= mc + hw_ws)
                    ))
                    m_sig_a    = (mass_arr_abcd >= mc - hw_ws) & (mass_arr_abcd <= mc + hw_ws)
                    n_mc_a_ws  = float(np.sum(weights_abcd[m_sig_a & z_sig_mask]))
                    m_sig_n    = (mass_arr_norm >= mc - hw_ws) & (mass_arr_norm <= mc + hw_ws)
                    n_mc_win_ws= float(np.sum(weights_norm[m_sig_n]))
                    if n_mc_win_ws > 0 and n_data_ws > 0:
                        rad_frac_val   = _eval_rad_frac(rad_frac, mc)
                        n_exp_ws       = sarah_prompt_yield(mass_hyp, eps2_val, n_data_ws, delta_m_ws, rad_frac_val)
                        nsig_ws_arr[i] = n_mc_a_ws * (n_exp_ws / n_mc_win_ws)
                ws_sum = float(np.sum(nsig_ws_arr))
                ws_results.append({"hw_GeV": hw_ws, "nsig": ws_sum})
                logger.info("    window_scan  hw=%.2f MeV → nsig=%.4g", hw_ws * 1000, ws_sum)

        results.append({
            "label":        label,
            "ap_mass":      ap_mass,
            "eps2":         eps2_val,
            "n_mc_a":       n_mc_a,
            "n_mc_total":   n_mc_total,
            "nsig":         nsig,
            "mass_centers": mass_centers,
            "norm_region":  norm_name,
            "window_scan":  ws_results,
        })

    return results


def plot_signal_2d(signal_scan_cache, output_dir, output_format):
    """2D heat map of expected signal yield N_sig(mass, ε²).

    For each region in signal_scan_cache, builds a grid over unique (ap_mass, eps2)
    pairs and plots log10(nsig) as a pcolormesh with contour lines at nsig = 0.1,
    1, 10, 100.
    """
    for region_name, scan_entries in signal_scan_cache.items():
        valid = [e for e in scan_entries
                 if e.get("ap_mass") is not None and e.get("eps2") is not None]
        if not valid:
            continue

        masses_sorted = sorted(set(e["ap_mass"] for e in valid))
        eps2_sorted   = sorted(set(e["eps2"]    for e in valid))
        if len(masses_sorted) < 2 or len(eps2_sorted) < 2:
            logger.warning("signal_2d: not enough (mass, ε²) points for region '%s' — skipping.",
                           region_name)
            continue

        masses_mev = np.array(masses_sorted) * 1000.0
        eps2_arr   = np.array(eps2_sorted)
        log_e      = np.log10(eps2_arr)

        # Build grid: rows = eps2 (ascending), cols = mass (ascending)
        # Use only the on-mass bin: the nsig element whose mass center is
        # closest to entry["ap_mass"].  Summing over all mass centers would
        # mix N_data from neighbouring bins (wrong physics).
        nsig_grid = np.full((len(eps2_sorted), len(masses_sorted)), np.nan)
        for entry in valid:
            im = masses_sorted.index(entry["ap_mass"])
            ie = eps2_sorted.index(entry["eps2"])
            mc_arr = np.asarray(entry.get("mass_centers", []), dtype=float)
            if len(mc_arr) > 0 and entry["ap_mass"] is not None:
                i_ap = int(np.argmin(np.abs(mc_arr - entry["ap_mass"])))
                nsig_grid[ie, im] = float(entry["nsig"][i_ap])
            else:
                nsig_grid[ie, im] = float(np.sum(entry["nsig"]))

        # pcolormesh bin edges
        dm = np.diff(masses_mev)
        m_edges = np.concatenate([[masses_mev[0] - dm[0] / 2],
                                   masses_mev[:-1] + dm / 2,
                                   [masses_mev[-1] + dm[-1] / 2]])
        de = np.diff(log_e)
        e_edges = np.concatenate([[log_e[0] - de[0] / 2],
                                   log_e[:-1] + de / 2,
                                   [log_e[-1] + de[-1] / 2]])

        log_nsig = np.where(nsig_grid > 0, np.log10(nsig_grid), np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(m_edges, e_edges, log_nsig,
                            cmap="viridis", vmin=-2, vmax=3)

        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label(r"$\log_{10}(N_{\rm sig})$", fontsize=11)
        cb.set_ticks([-2, -1, 0, 1, 2, 3])
        cb.set_ticklabels(["0.01", "0.1", "1", "10", "100", "1000"])

        # Contour lines at nsig = 0.1, 1, 10, 100
        contour_levels = [-1, 0, 1, 2]   # log10 of [0.1, 1, 10, 100]
        contour_labels = {-1: "0.1", 0: "1", 1: "10", 2: "100"}
        try:
            cs = ax.contour(masses_mev, log_e, log_nsig,
                            levels=contour_levels,
                            colors=["white", "white", "white", "white"],
                            linewidths=[0.8, 2.0, 0.8, 0.8],
                            linestyles=["--", "-", "--", "--"])
            ax.clabel(cs, fmt=contour_labels, fontsize=9, inline=True)
        except Exception as exc:
            logger.debug("signal_2d contour failed for region '%s': %s", region_name, exc)

        ax.set_xlabel(r"$m_{A'}$ [MeV]", fontsize=12)
        ax.set_ylabel(r"$\varepsilon^2$", fontsize=12)
        ax.set_yticks(log_e)
        ax.set_yticklabels(
            [f"$10^{{{int(v)}}}$" if v == int(v) else f"$10^{{{v:.1f}}}$" for v in log_e]
        )
        ax.set_title(f"Signal yield $N_{{\\rm sig}}$ — {region_name}", fontsize=12)

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        fname = f"signal_2d_{region_name}.{output_format}"
        fig.savefig(outdir / fname, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved 2D signal plot: %s", outdir / fname)


def plot_data_yield_2d(signal_scan_cache, data_yield_cache, output_dir, output_format):
    """2D diagnostic: data yield dN/dm in signal mass window vs (mass, ε²).

    Builds the same (mass, ε²) grid as plot_signal_2d but fills every cell with
    dN/dm = N_data_window / (2·hw).  Since data yield is independent of ε², every
    row should be identical — this is the cross-check that the Eq. 4 normalization
    is not accidentally ε²-dependent.
    """
    for region_name, scan_entries in signal_scan_cache.items():
        if region_name not in data_yield_cache:
            continue

        yield_info = data_yield_cache[region_name]
        n_data = yield_info["n_data_mass_only"]
        mass_centers = yield_info["mass_centers"]
        abcd_cfg = yield_info["abcd_cfg"]

        valid = [e for e in scan_entries
                 if e.get("ap_mass") is not None and e.get("eps2") is not None]
        if not valid:
            continue

        eps2_sorted = sorted(set(e["eps2"] for e in valid))
        if len(eps2_sorted) < 2:
            logger.warning("data_yield_2d: not enough ε² points for region '%s' — skipping.",
                           region_name)
            continue

        # Eq. 4 data factor: N_data × m / δm — window-independent to first order.
        # N_data ≈ f(m) × δm for smooth spectrum, so N_data × m / δm ≈ f(m) × m
        # which is independent of the signal window half-width hw.
        hw_arr = np.array([_compute_window(abcd_cfg, region_name, mc)[0] for mc in mass_centers])
        eq4_data = np.where(hw_arr > 0, n_data * mass_centers / (2.0 * hw_arr), 0.0)

        masses_mev = mass_centers * 1000.0
        log_e = np.log10(np.array(eps2_sorted))

        # Same value for every eps2 row (data yield is eps2-independent)
        eq4_grid = np.outer(np.ones(len(eps2_sorted)), eq4_data)

        # pcolormesh bin edges
        dm = np.diff(masses_mev)
        m_edges = np.concatenate([[masses_mev[0] - dm[0] / 2],
                                   masses_mev[:-1] + dm / 2,
                                   [masses_mev[-1] + dm[-1] / 2]])
        de = np.diff(log_e)
        e_edges = np.concatenate([[log_e[0] - de[0] / 2],
                                   log_e[:-1] + de / 2,
                                   [log_e[-1] + de[-1] / 2]])

        log_eq4 = np.where(eq4_grid > 0, np.log10(eq4_grid), np.nan)
        vmin = np.nanmin(log_eq4) if not np.all(np.isnan(log_eq4)) else 0.0
        vmax = np.nanmax(log_eq4) if not np.all(np.isnan(log_eq4)) else 1.0

        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(m_edges, e_edges, log_eq4,
                            cmap="plasma", vmin=vmin, vmax=vmax)

        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label(r"$\log_{10}(N_{\rm data} \cdot m\,/\,\delta m)$ [events]", fontsize=11)

        ax.set_xlabel(r"$m_{A'}$ [MeV]", fontsize=12)
        ax.set_ylabel(r"$\varepsilon^2$", fontsize=12)
        ax.set_yticks(log_e)
        ax.set_yticklabels(
            [f"$10^{{{int(v)}}}$" if v == int(v) else f"$10^{{{v:.1f}}}$" for v in log_e]
        )
        ax.set_title(
            f"Eq. 4 data factor $N_{{\\rm data}}\\cdot m/\\delta m$ — {region_name}\n"
            r"(window-independent to first order; rows should be identical)",
            fontsize=11,
        )

        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        fname = f"data_yield_2d_{region_name}.{output_format}"
        fig.savefig(outdir / fname, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved 2D data yield plot: %s", outdir / fname)


def plot_signal_window_scan(signal_scan_cache, region_name, output_dir, output_format,
                            nominal_hw=None):
    """Plot nsig vs mass-window half-width for all signal samples in one figure.

    One curve per (sample, eps2) entry.  y-axis is nsig normalised to the value
    at the nominal window so all curves start at 1.0 and deviations are visible
    as a fraction.  A vertical dashed line marks the nominal hw when supplied.
    """
    scan_entries = signal_scan_cache.get(region_name, [])
    # Keep only entries that have a populated window_scan list
    plottable = [e for e in scan_entries if e.get("window_scan")]
    if not plottable:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    cmap = plt.get_cmap("tab10")
    for idx, entry in enumerate(plottable):
        ws = entry["window_scan"]
        hw_vals  = np.array([p["hw_GeV"] * 1000 for p in ws])   # → MeV
        nsig_vals = np.array([p["nsig"] for p in ws])

        # Normalise to the value closest to the nominal hw
        if nominal_hw is not None:
            i_nom = int(np.argmin(np.abs(hw_vals - nominal_hw * 1000)))
        else:
            i_nom = len(hw_vals) // 2
        ref = nsig_vals[i_nom]
        if ref > 0:
            nsig_norm = nsig_vals / ref
        else:
            nsig_norm = nsig_vals

        m_GeV = entry.get("ap_mass")
        m_label = f"{m_GeV * 1000:.0f} MeV" if m_GeV is not None else entry["label"]

        ax.plot(hw_vals, nsig_norm, marker="o", ms=4, lw=1.5,
                color=cmap(idx % 10), label=m_label)

    if nominal_hw is not None:
        ax.axvline(nominal_hw * 1000, color="gray", lw=1, ls="--",
                   label=f"nominal hw = {nominal_hw * 1000:.1f} MeV")

    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Signal window half-width [MeV]")
    ax.set_ylabel("nsig / nsig(nominal)")
    ax.set_title(f"Window-size systematic — {region_name}")
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"window_scan_{region_name}.{output_format}"
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outdir / fname)


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


def _build_mass_centers(abcd_cfg, plot_cfg, samples_map):
    """Return (mass_centers, step_arr) for the mass scan.

    When abcd_cfg.mass_scan_from_signal_samples is True, mass_centers are the
    sorted unique ap_mass values of signal samples within [mass_scan_min,
    mass_scan_max].  step_arr gives the per-bin midpoint spacing, used by
    _integrate_region.

    When False (default), a regular arange grid is used with uniform step.
    """
    if abcd_cfg.mass_scan_from_signal_samples:
        masses = sorted({
            samples_map[n].ap_mass
            for n in plot_cfg.samples
            if samples_map[n].sample_type == "signal"
            and samples_map[n].ap_mass is not None
            and abcd_cfg.mass_scan_min <= samples_map[n].ap_mass <= abcd_cfg.mass_scan_max
        })
        if masses:
            mc = np.array(masses)
            if len(mc) == 1:
                step_arr = np.array([abcd_cfg.mass_scan_step])
            else:
                diffs    = np.diff(mc)
                step_arr = np.concatenate([
                    [diffs[0]],
                    (diffs[:-1] + diffs[1:]) / 2.0,
                    [diffs[-1]],
                ])
            logger.info(
                "mass_scan_from_signal_samples: %d points — %s MeV",
                len(mc), [f"{m * 1000:.0f}" for m in mc],
            )
            return mc, step_arr
        logger.warning(
            "mass_scan_from_signal_samples=True but no signal samples have "
            "ap_mass in [%.3f, %.3f] GeV — falling back to regular grid.",
            abcd_cfg.mass_scan_min, abcd_cfg.mass_scan_max,
        )

    step = abcd_cfg.mass_scan_step
    mc   = np.arange(
        abcd_cfg.mass_scan_min,
        abcd_cfg.mass_scan_max + step * 0.5,
        step,
    )
    return mc, np.full(len(mc), step)


def _integrate_region(counts_dict, mass_centers, step_arr, abcd_cfg, region_name):
    """Deweight sliding-window counts to produce a proper event integral.

    Each window of half-width hw_i centred on mass_centers[i] contains the
    same event ~(2*hw_i / step_i) times across neighbouring windows.  The factor
    w_i = step_i / (2*hw_i) converts per-window counts back to a per-GeV density
    summed over step_i → total events in the scanned mass range.

    step_arr may be a scalar (uniform grid) or a 1-D array (signal-sample mode).

    Returns
    -------
    na_total, na_err_total, bkg_total, bkg_err_total : float
    """
    hw_arr = np.array(
        [_compute_window(abcd_cfg, region_name, mc)[0] for mc in mass_centers]
    )
    dw = np.asarray(step_arr) / (2.0 * hw_arr)

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
    mass_centers, step_arr = _build_mass_centers(abcd_cfg, plot_cfg, samples_map)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_names = [s for s in plot_cfg.samples if samples_map[s].sample_type == "data"]
    mc_names   = [s for s in plot_cfg.samples if samples_map[s].sample_type not in ("data", "signal")]

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
            mc_na_dict, mass_centers, step_arr, abcd_cfg, region_name
        )
        mc_vals.append(mc_tot)
        mc_errs.append(mc_tot_err)

        # Data observed and ABCD estimate
        obs_tot = obs_err2 = 0.0
        bkg_tot = bkg_err2 = 0.0
        for name in data_names:
            na_t, na_e, bg_t, bg_e = _integrate_region(
                counts[name], mass_centers, step_arr, abcd_cfg, region_name
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
    add_hps_label(ax, lumi=config.luminosity, run_label=config.run_label)

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
             ann_scorer=None, data_cache=None, all_region_branches=None,
             signal_scan_cache=None, data_yield_cache=None):
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

    mass_centers, _step_arr = _build_mass_centers(abcd_cfg, plot_cfg, samples_map)

    # --- log region and ABCD zone definitions --------------------------------
    # Show the full selection string for the region and each ABCD zone so the
    # user can verify exactly what cuts are applied.
    logger.info(
        "=== ABCD: plot='%s'  region='%s' ===\n"
        "  Base selection : %s\n"
        "  Mass variable  : %s\n"
        "  Z variable     : %s\n"
        "  Mass scan      : %.1f – %.1f MeV  (step %.1f MeV, %d bins)\n"
        "  Window sizing  : sr=%.1fσ  gap=%.1fσ  sb=%.1fσ  (fixed fallback hw=%.4f gap=%.4f sw=%.4f GeV)\n"
        "  Z signal  (A,C): %s > %.3f mm  and  %s < %.3g mm\n"
        "  Z control (B,D): %s >= %.3f mm  and  %s <= %.3f mm\n"
        "  Zone definitions (for a given mass hypothesis m, window hw=%.1fσ·σ(m)):\n"
        "    A  (signal region)   : (%s) AND |m_ee - m_hyp| < hw AND %s > %.3f\n"
        "    B  (z-control)       : (%s) AND |m_ee - m_hyp| < hw AND %.3f <= %s <= %.3f\n"
        "    C  (mass sideband, z-sig)  : (%s) AND m_ee in sideband AND %s > %.3f\n"
        "    D  (mass sideband, z-ctrl) : (%s) AND m_ee in sideband AND %.3f <= %s <= %.3f\n"
        "  Estimate: N_A^bkg = N_B * N_C / N_D",
        plot_cfg.name, region_name,
        region_cfg.selection,
        abcd_cfg.mass_variable,
        abcd_cfg.z_variable,
        abcd_cfg.mass_scan_min * 1000, abcd_cfg.mass_scan_max * 1000,
        abcd_cfg.mass_scan_step * 1000, len(mass_centers),
        abcd_cfg.sr_sigmas, abcd_cfg.gap_sigmas, abcd_cfg.sb_sigmas,
        abcd_cfg.mass_window_half_width, abcd_cfg.sideband_gap, abcd_cfg.sideband_width,
        abcd_cfg.z_variable, abcd_cfg.z_signal_min,
        abcd_cfg.z_variable, abcd_cfg.z_signal_max,
        abcd_cfg.z_variable, abcd_cfg.z_control_min,
        abcd_cfg.z_variable, abcd_cfg.z_control_max,
        abcd_cfg.sr_sigmas,
        region_cfg.selection, abcd_cfg.z_variable, abcd_cfg.z_signal_min,
        region_cfg.selection, abcd_cfg.z_control_min, abcd_cfg.z_variable, abcd_cfg.z_control_max,
        region_cfg.selection, abcd_cfg.z_variable, abcd_cfg.z_signal_min,
        region_cfg.selection, abcd_cfg.z_control_min, abcd_cfg.z_variable, abcd_cfg.z_control_max,
    )
    if abcd_cfg.mass_resolution:
        for rname, expr in abcd_cfg.mass_resolution.items():
            logger.info("  Mass resolution [%s]: σ(m) = %s", rname, expr)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- classify samples ---------------------------------------------------
    data_names   = [s for s in plot_cfg.samples if samples_map[s].sample_type == "data"]
    mc_names     = [s for s in plot_cfg.samples if samples_map[s].sample_type not in ("data", "signal")]
    signal_names = [s for s in plot_cfg.samples if samples_map[s].sample_type == "signal"]

    # --- load counts per sample ---------------------------------------------
    # When data_cache is provided (shared across region calls) each sample is
    # loaded from disk only once; subsequent regions reuse the cached arrays
    # and just re-apply the region mask via _count_from_raw.
    counts = {}
    for name in mc_names + signal_names + data_names:
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
    add_hps_label(ax, lumi=config.luminosity, extra_lines=[region_cfg.label], run_label=config.run_label)

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

    # ---- signal weight_scan: Eq. 4 scaled yields per eps² -----------------
    if signal_scan_cache is not None:
        # Resolve optional normalization region (loose reference for Eq. 4)
        norm_region_obj = None
        if abcd_cfg.signal_norm_region:
            norm_region_cfg = next(
                (r for r in config.regions if r.name == abcd_cfg.signal_norm_region), None
            )
            if norm_region_cfg is not None:
                norm_region_obj = Region(norm_region_cfg)
                logger.info(
                    "Signal scan norm region: '%s' (overrides ABCD region '%s')",
                    abcd_cfg.signal_norm_region, region_name,
                )
            else:
                logger.warning(
                    "signal_norm_region '%s' not found in config — "
                    "falling back to ABCD region '%s'.",
                    abcd_cfg.signal_norm_region, region_name,
                )

        # N_data in mass window, no z cut — from the chosen norm region
        n_data_mass_only = np.zeros(len(mass_centers))
        if norm_region_obj is not None and data_cache is not None:
            for name in data_names:
                if name in data_cache:
                    c = _count_from_raw(
                        data_cache[name], norm_region_obj,
                        samples_map[name], mass_centers, abcd_cfg,
                    )
                    n_data_mass_only += c.get("n_mass_only", np.zeros(len(mass_centers)))
        else:
            for name in data_names:
                n_data_mass_only += counts[name].get("n_mass_only", np.zeros(len(mass_centers)))

        # If signal_norm_hw is set, recount n_data_mass_only with that window.
        if abcd_cfg.signal_norm_hw is not None and data_cache is not None:
            snhw = abcd_cfg.signal_norm_hw
            eff_norm = norm_region_obj if norm_region_obj is not None else region
            n_data_mass_only = np.zeros(len(mass_centers))
            for name in data_names:
                if name not in data_cache:
                    continue
                dr    = data_cache[name]
                dmask = eff_norm.apply(dr["data"])
                cfg_s = samples_map[name]
                if cfg_s.selection:
                    dmask = dmask & np.asarray(
                        safe_evaluate(cfg_s.selection, dr["data"]), dtype=bool
                    )
                mass_a = np.asarray(
                    safe_evaluate(abcd_cfg.mass_variable, dr["data"], mask=dmask), dtype=float
                )
                for i, mc in enumerate(mass_centers):
                    n_data_mass_only[i] += float(
                        np.sum((mass_a >= mc - snhw) & (mass_a <= mc + snhw))
                    )
            logger.info(
                "Signal norm window: signal_norm_hw=%.4f GeV  (delta_m=%.4f GeV)",
                snhw, 2.0 * snhw,
            )

        if data_yield_cache is not None:
            data_yield_cache[region_name] = {
                "n_data_mass_only": n_data_mass_only.copy(),
                "mass_centers": mass_centers.copy(),
                "abcd_cfg": abcd_cfg,
            }

        # Data mass array for window-size scan (extracted once, reused across signal samples)
        data_mass_arr_for_scan = None
        if abcd_cfg.window_scan and data_cache is not None and data_names:
            eff_norm_ws = norm_region_obj if norm_region_obj is not None else region
            arrs = []
            for name in data_names:
                if name in data_cache:
                    dr = data_cache[name]
                    dmask = eff_norm_ws.apply(dr["data"])
                    cfg_s = samples_map[name]
                    if cfg_s.selection:
                        dmask = dmask & np.asarray(
                            safe_evaluate(cfg_s.selection, dr["data"]), dtype=bool
                        )
                    arrs.append(np.asarray(
                        safe_evaluate(abcd_cfg.mass_variable, dr["data"], mask=dmask),
                        dtype=float,
                    ))
            if arrs:
                data_mass_arr_for_scan = np.concatenate(arrs)

        scan_results = []
        for name in signal_names:
            cfg_s = samples_map[name]
            if not cfg_s.weight_scan and cfg_s.eps2_scan is None:
                continue
            raw = data_cache[name] if data_cache is not None else None
            if raw is None:
                logger.warning("Signal scan: no cached data for '%s' — skipping.", name)
                continue
            scan_results.extend(
                _count_signal_scan_from_raw(
                    raw, region, cfg_s, mass_centers, abcd_cfg,
                    config.scaling_rad_frac, n_data_mass_only,
                    norm_region=norm_region_obj,
                    data_mass_arr_norm=data_mass_arr_for_scan,
                )
            )

        signal_scan_cache[region_name] = scan_results

        # SIMP diagnostic plot: nsig vs ε² overlaid with A', plus ratio panel
        simp_entries = [r for r in scan_results if "simp" in r.get("label", "")]
        ap_entries   = [r for r in scan_results if "simp" not in r.get("label", "")]
        if simp_entries:
            simp_scaling.plot_simp_diagnostics(
                simp_entries, ap_entries or None,
                region_name, outdir, output_format,
            )

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

    mass_centers, step_arr = _build_mass_centers(abcd_cfg, plot_cfg, samples_map)

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
        add_hps_label(ax, lumi=L_ref, extra_lines=[region_cfg_obj.label], run_label=config.run_label)

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


def write_abcd_json(plot_cfg, config, samples_map, counts_cache, output_path,
                    signal_scan_cache=None):
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

    mass_centers, _step_arr = _build_mass_centers(abcd_cfg, plot_cfg, samples_map)

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

            if signal_scan_cache and region_name in signal_scan_cache:
                entry["signal"] = {}
                for sr in signal_scan_cache[region_name]:
                    sig_entry = {
                        "epsilon_sq":  sr["eps2"],
                        "nsig":        round(float(sr["nsig"][i]), 6),
                        "n_mc_a":      round(float(sr["n_mc_a"][i]), 6),
                        "acceptance":  round(float(sr["n_mc_a"][i] / sr["n_mc_total"]), 8)
                                       if sr["n_mc_total"] > 0 else 0.0,
                        "norm_region": sr["norm_region"],
                    }
                    if sr.get("window_scan"):
                        sig_entry["window_scan"] = sr["window_scan"]
                    entry["signal"][sr["label"]] = sig_entry

            mass_points.append(entry)

        regions_data[region_name] = mass_points

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "regions": regions_data}, f, indent=2)
    logger.info("ABCD results JSON written: %s", out_path)


def plot_abcd_aux_histogram(plot_cfg, aux_cfg, base_region_cfg, bin_region_names,
                             bin_region_cfgs, config, samples_map, output_dir, output_format,
                             counts_cache=None, data_cache=None):
    """Histogram of integrated ABCD estimate as a function of an aux variable.

    Each element of *bin_region_names* is a child region that covers one bin of
    ``aux_cfg.bins``.  For each bin the ABCD counts (already computed and cached
    in *counts_cache*) are integrated over the mass scan to produce a single
    event count.

    All three quantities — ABCD estimate, observed data, MC sum — are drawn
    using the **region-A** definition (signal mass window AND signal z region),
    so the comparison is apples-to-apples:

    Top panel
    ---------
    * ABCD estimate — dashed red step + uncertainty band (from data sidebands)
    * MC (shape)   — solid navy step, no z/mass cuts, normalised to total ABCD
                     estimate so the shape can be compared without being zero.
                     Relative contributions of individual MC processes are
                     preserved before the overall normalisation is applied.
    * Observed data — black error-bar points at bin centres (region A observed)

    Bottom panel
    ------------
    * ABCD / obs.  (red circles)
    * MC / obs.    (navy squares, after normalisation; only when MC present)
    """
    abcd_cfg = plot_cfg.abcd
    edges   = np.array(aux_cfg.bins, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins  = len(centers)

    if n_bins == 0 or n_bins != len(bin_region_names):
        logger.warning(
            "plot_abcd_aux_histogram: mismatch between aux bins (%d) and "
            "bin_region_names (%d) — skipping.",
            n_bins, len(bin_region_names),
        )
        return

    mass_centers, _step_arr = _build_mass_centers(abcd_cfg, plot_cfg, samples_map)

    data_names   = [s for s in plot_cfg.samples if samples_map[s].sample_type == "data"]
    bkg_names    = [s for s in plot_cfg.samples if samples_map[s].sample_type == "background"]
    signal_names = [s for s in plot_cfg.samples if samples_map[s].sample_type == "signal"]
    has_data   = bool(data_names)
    has_bkg    = bool(bkg_names)
    has_signal = bool(signal_names)

    obs_vals  = np.zeros(n_bins)
    obs_errs  = np.zeros(n_bins)
    bkg_vals  = np.zeros(n_bins)
    bkg_errs2 = np.zeros(n_bins)
    # Background MC: lumi-weighted counts with ONLY the bin region mask applied
    # (no z/mass cuts). Normalised to total ABCD; relative contributions preserved.
    bkg_mc_raw = np.zeros(n_bins)
    # Signal: one raw array per sample, each normalised independently to ABCD total.
    sig_raw = {name: np.zeros(n_bins) for name in signal_names}

    def _region_count(name, region_obj, data_arr_dict):
        """Lumi-weighted event count in region_obj for sample *name* (no z/mass cuts)."""
        raw         = data_arr_dict[name]
        data_arr    = raw["data"]
        total_scale = raw["total_scale"]
        weight_expr = raw["weight_expr"]
        cfg_s       = samples_map[name]

        mask = region_obj.apply(data_arr)
        if cfg_s.selection:
            mask = mask & np.asarray(safe_evaluate(cfg_s.selection, data_arr), dtype=bool)

        raw_w = safe_evaluate(weight_expr, data_arr, mask=mask)
        if np.ndim(raw_w) == 0:
            return float(np.sum(mask)) * float(raw_w) * total_scale
        else:
            return float(np.sum(np.asarray(raw_w, dtype=float))) * total_scale

    for i, (rname, region_cfg_i) in enumerate(zip(bin_region_names, bin_region_cfgs)):
        if not counts_cache or rname not in counts_cache:
            logger.warning("Aux histogram: counts not found for region '%s', skipping bin %d.", rname, i)
            continue
        counts = counts_cache[rname]

        # --- data: observed (na) and ABCD estimate (from nb, nc, nd) --------
        for name in data_names:
            if name not in counts:
                continue
            na_t, na_e, bg_t, bg_e = _integrate_region(
                counts[name], mass_centers, step_arr, abcd_cfg, rname
            )
            obs_vals[i]  += na_t
            obs_errs[i]   = sqrt(obs_errs[i] ** 2 + na_e ** 2)
            bkg_vals[i]  += bg_t
            bkg_errs2[i] += bg_e ** 2

        # --- background MC and signal: region-mask-only counts --------------
        if data_cache is not None:
            region_obj = Region(region_cfg_i)
            for name in bkg_names:
                if name in data_cache:
                    bkg_mc_raw[i] += _region_count(name, region_obj, data_cache)
            for name in signal_names:
                if name in data_cache:
                    sig_raw[name][i] += _region_count(name, region_obj, data_cache)

    bkg_errs   = np.sqrt(bkg_errs2)
    total_abcd = float(np.sum(bkg_vals))

    # Normalise background MC sum to total ABCD (preserves relative shape)
    total_bkg_mc = float(np.sum(bkg_mc_raw))
    if total_bkg_mc > 0 and total_abcd > 0:
        bkg_mc_vals = bkg_mc_raw * (total_abcd / total_bkg_mc)
    else:
        bkg_mc_vals = bkg_mc_raw.copy()
        if has_bkg and total_bkg_mc == 0:
            logger.warning("Aux histogram: all background MC bins are zero before normalisation.")

    # Normalise each signal sample independently to total ABCD
    sig_vals = {}
    for name in signal_names:
        total_sig = float(np.sum(sig_raw[name]))
        if total_sig > 0 and total_abcd > 0:
            sig_vals[name] = sig_raw[name] * (total_abcd / total_sig)
        else:
            sig_vals[name] = sig_raw[name].copy()

    # --- figure -------------------------------------------------------
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    label_lines = []
    if base_region_cfg is not None:
        lbl = base_region_cfg.label or base_region_cfg.name
        if lbl:
            label_lines = [lbl]
    add_hps_label(ax, lumi=config.luminosity, extra_lines=label_lines, run_label=config.run_label)

    # ABCD estimate — dashed red step
    ax.step(centers, bkg_vals, where="mid",
            color="red", lw=2, ls="--", label="ABCD estimate")
    ax.fill_between(
        centers,
        np.maximum(bkg_vals - bkg_errs, 0), bkg_vals + bkg_errs,
        step="mid", color="red", alpha=0.10,
    )

    # Background MC — solid navy step, normalised to total ABCD (shape only)
    if has_bkg:
        bkg_label = (
            (" + ".join(samples_map[n].label for n in bkg_names) if len(bkg_names) <= 3 else "MC bkg.")
            + " (norm. to ABCD)"
        )
        ax.step(centers, bkg_mc_vals, where="mid",
                color="navy", lw=2, label=bkg_label)

    # Signal samples — each individually normalised to total ABCD, dashed lines
    for name in signal_names:
        cfg_s = samples_map[name]
        ax.step(centers, sig_vals[name], where="mid",
                color=cfg_s.color, lw=1.5, ls="--",
                label=f"{cfg_s.label} (norm. to ABCD)")

    # Observed data — black error bars
    if has_data:
        ax.errorbar(
            centers, obs_vals, yerr=obs_errs,
            fmt="o", color="black", markersize=5, zorder=5,
            label="Data (obs.)",
        )

    ax.set_ylabel("Events (integrated)")
    if abcd_cfg.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * (10 if abcd_cfg.log_y else 1.5))
    ax.legend(fontsize=10, loc="upper right")

    # --- ratio panel --------------------------------------------------
    ax_r.axhline(1.0, color="gray", lw=1, ls="--")

    if has_data:
        valid = obs_vals > 0
        ratio_abcd     = np.where(valid, bkg_vals / obs_vals, np.nan)
        ratio_abcd_err = np.where(valid, bkg_errs / obs_vals, np.nan)
        ax_r.errorbar(
            centers, ratio_abcd, yerr=ratio_abcd_err,
            fmt="o", color="red", markersize=4, label="ABCD / obs.",
        )

    if has_bkg and has_data:
        ratio_bkg = np.where(valid, bkg_mc_vals / obs_vals, np.nan)
        ax_r.step(centers, ratio_bkg, where="mid",
                  color="navy", lw=2, label="MC bkg. / obs.")

    ax_r.set_xlabel(aux_cfg.label or aux_cfg.variable)
    ax_r.set_ylabel("Ratio")
    ax_r.set_ylim(0, 3)
    ax_r.set_xlim(edges[0], edges[-1])
    if has_data or has_bkg:
        ax_r.legend(fontsize=9, loc="upper right")

    # Sanitise the variable name for use in the filename
    var_clean = _re.sub(r'[^a-zA-Z0-9]', '_', aux_cfg.variable).strip('_')
    base_name = base_region_cfg.name if base_region_cfg is not None else "base"
    fname = f"{plot_cfg.name}_{base_name}_{var_clean}.{output_format}"
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outdir / fname)
