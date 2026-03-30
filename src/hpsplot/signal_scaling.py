"""Data-driven signal scaling via Eq. 4 of PhysRevD.108.012015.

Eq. 4:  dσ_{A'}/dm = (3π m_{A'} ε²) / (2 N_eff α) × dσ_{γ*}/dm

With N_eff = 1 this gives the expected signal yield in a mass window as:

    N_expected = (3π m_{A'} ε²) / (2α) × N_data_in_window

The scale factor applied to the signal MC is:

    f = N_expected / N_mc_in_scaling_region

Usage in results.py:
    # Load data once
    n_data = count_data_in_window(data_cfg, region_cfg, mass_var, hw, aliases)
    # Per signal sample
    scale = compute_eq4_scale(signal_cfg, n_data, region_cfg, aliases)
"""

import logging

import numpy as np

from .region import Region
from .sample import Sample
from .utils import extract_branch_names, safe_evaluate

logger = logging.getLogger(__name__)

ALPHA_EM = 1.0 / 137.036  # fine structure constant (N_eff = 1)


def count_data_in_window(
    data_cfg,
    scaling_region_cfg,
    mass_variable,
    mass_window_half_width,
    ap_mass,
    global_aliases,
):
    """Count data events in [ap_mass ± mass_window_half_width] within the scaling region.

    Called once per run; the result is cached by the caller and passed to
    every :func:`compute_eq4_scale` call.

    Returns
    -------
    float
        N_data_window — raw event count (weight = 1 for data).
    """
    data_sample = Sample(data_cfg)
    scaling_region = Region(scaling_region_cfg)

    eff_aliases = {**global_aliases, **data_cfg.aliases}
    needed = (
        extract_branch_names(mass_variable)
        | extract_branch_names(scaling_region_cfg.selection)
    )
    if data_cfg.selection:
        needed |= extract_branch_names(data_cfg.selection)

    arrays = data_sample.load(list(needed), aliases=eff_aliases)

    region_mask = scaling_region.apply(arrays)
    if data_cfg.selection:
        region_mask = region_mask & np.asarray(
            safe_evaluate(data_cfg.selection, arrays), dtype=bool
        )

    mass_values = safe_evaluate(mass_variable, arrays, mask=region_mask)
    n_data = float(np.sum(np.abs(mass_values - ap_mass) < mass_window_half_width))

    logger.info(
        "Eq. 4 — data: %d events in [%.4f, %.4f] GeV (region '%s')",
        int(n_data),
        ap_mass - mass_window_half_width,
        ap_mass + mass_window_half_width,
        scaling_region_cfg.name,
    )
    return n_data


def compute_eq4_scale(
    signal_cfg,
    n_data_window,
    scaling_region_cfg,
    global_aliases,
    mass_window_half_width,
    rad_frac,
):
    """Compute the Eq. 4 scale factor for one signal sample.

    Formula (PhysRevD.108.012015, Eq. 4):

        S_bin = f_rad × N_bin × (3π ε²) / (2 N_eff α) × m_A' / δm_A'

    where δm_A' = 2 × mass_window_half_width.

    Parameters
    ----------
    signal_cfg : SampleConfig
        Signal sample; must have ``ap_mass`` and ``epsilon_sq`` set.
    n_data_window : float
        Data event count N_bin in the mass window, from :func:`count_data_in_window`.
    scaling_region_cfg : RegionConfig
        Preselection region (``is_scaling_region = True``).
    global_aliases : dict
        Global aliases from the top-level config.
    mass_window_half_width : float
        Half-width hw [GeV] used to count N_bin; δm_A' = 2 × hw.
    rad_frac : float
        Radiative fraction f_rad (fraction of background from radiative processes).

    Returns
    -------
    float
        Scale factor *f* such that ``f × N_mc_scaling = S_bin``.
    """
    ap_mass = signal_cfg.ap_mass
    epsilon_sq = signal_cfg.epsilon_sq

    if ap_mass is None or epsilon_sq is None:
        raise ValueError(
            f"Signal sample '{signal_cfg.name}' needs ap_mass and epsilon_sq "
            "for Eq. 4 scaling."
        )

    # ------------------------------------------------------------------ #
    # Sum weighted signal MC yield in the scaling region                  #
    # ------------------------------------------------------------------ #
    signal_sample = Sample(signal_cfg)
    scaling_region = Region(scaling_region_cfg)
    eff_aliases = {**global_aliases, **signal_cfg.aliases}

    needed = extract_branch_names(scaling_region_cfg.selection)
    if signal_cfg.selection:
        needed |= extract_branch_names(signal_cfg.selection)
    needed |= extract_branch_names(signal_cfg.weight)

    arrays = signal_sample.load(list(needed), aliases=eff_aliases)

    region_mask = scaling_region.apply(arrays)
    if signal_cfg.selection:
        region_mask = region_mask & np.asarray(
            safe_evaluate(signal_cfg.selection, arrays), dtype=bool
        )

    weights = safe_evaluate(signal_cfg.weight, arrays, mask=region_mask)
    if np.ndim(weights) == 0:
        weights = np.full(int(np.sum(region_mask)), float(weights))

    # Guard against overflow: events with truth vtx_z < 0 produce exp(+|z|/L)
    # which diverges for small cτ.  These events are unphysical (A' decayed
    # upstream) and should contribute zero weight.
    n_raw = len(weights)
    n_inf = int(np.sum(~np.isfinite(weights)))
    n_neg = int(np.sum(weights[np.isfinite(weights)] < 0.0))
    logger.debug(
        "Eq. 4 — signal MC '%s' weight diagnostics: "
        "n_events=%d, n_inf/nan=%d, n_negative=%d, "
        "w_min=%.4g, w_max=%.4g, w_mean=%.4g",
        signal_cfg.name, n_raw, n_inf, n_neg,
        float(np.nanmin(weights)) if n_raw > 0 else 0,
        float(np.nanmax(weights)) if n_raw > 0 else 0,
        float(np.nanmean(weights)) if n_raw > 0 else 0,
    )

    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.where(weights >= 0.0, weights, 0.0)

    n_mc_scaling = float(np.sum(weights))
    logger.debug(
        "Eq. 4 — signal MC '%s' after clipping: "
        "n_nonzero=%d / %d, sum=%.4g",
        signal_cfg.name,
        int(np.sum(weights > 0)), n_raw, n_mc_scaling,
    )

    logger.info(
        "Eq. 4 — signal MC '%s': weighted yield in scaling region '%s' = %.4g",
        signal_cfg.name, scaling_region_cfg.name, n_mc_scaling,
    )

    if n_mc_scaling == 0.0:
        logger.warning(
            "Signal MC '%s' has zero yield in scaling region '%s' — scale factor = 0.",
            signal_cfg.name, scaling_region_cfg.name,
        )
        return 0.0

    # ------------------------------------------------------------------ #
    # Apply Eq. 4:  S_bin = f_rad × N_bin × (3πε²)/(2α) × m_A'/δm_A'   #
    # ------------------------------------------------------------------ #
    delta_m = 2.0 * mass_window_half_width
    n_signal_expected = (
        rad_frac
        * n_data_window
        * (3.0 * np.pi * epsilon_sq) / (2.0 * ALPHA_EM)
        * ap_mass / delta_m
    )
    scale_factor = n_signal_expected / n_mc_scaling

    logger.info(
        "Eq. 4 result for '%s': m_A'=%.4f GeV, ε²=%.4g, "
        "f_rad=%.3f, δm=%.4f GeV, N_bin=%.0f, N_mc=%.4g, "
        "S_bin=%.4g → scale=%.4g",
        signal_cfg.name, ap_mass, epsilon_sq,
        rad_frac, delta_m, n_data_window, n_mc_scaling,
        n_signal_expected, scale_factor,
    )

    return scale_factor
