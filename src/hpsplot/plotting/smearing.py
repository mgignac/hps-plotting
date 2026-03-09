"""Smearing factor derivation and diagnostic plots."""

import logging
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .style import add_hps_label
from ..histogram import Histogram2DData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian fit utilities
# ---------------------------------------------------------------------------

def _gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _iterative_gauss_fit(centers, contents, errors, sigma_range=1.5, fit_range=None, n_iter=3,
                         core_fraction=None):
    """Fit a Gaussian iteratively, refining the range to [mu ± sigma_range*sigma].

    The window never narrows below 2 bin widths on each side, so fits on
    distributions narrower than the bin width (e.g. tight MC z0) still converge.

    Parameters
    ----------
    core_fraction : float or None
        When set (e.g. 0.70), the initial fit window is seeded from the central
        *core_fraction* of the weighted histogram (percentile-based), and
        sigma_range is derived from the corresponding Gaussian quantile
        (≈ 1.04 for 0.70).  This restricts the fit to the Gaussian core and
        suppresses non-Gaussian tails automatically.  When None, the standard
        sigma_range window is used from the first iteration.

    Returns (mu, sigma, mu_err, sigma_err) or None on failure.
    """
    if fit_range is not None:
        mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    else:
        mask = np.ones(len(centers), dtype=bool)

    # Minimum half-width: 2 bin widths — guarantees ≥ 4 bins in the window
    bin_width = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
    min_half_width = 2.0 * bin_width

    # Derive sigma_range and seed the initial window from data percentiles
    if core_fraction is not None:
        from scipy.special import erfinv as _erfinv
        # Gaussian quantile: P(|X-mu| < k*sigma) = erf(k/sqrt(2)) = core_fraction
        sigma_range = float(_erfinv(core_fraction) * np.sqrt(2))

        # Seed initial window from weighted percentiles of the distribution
        c_masked = centers[mask]
        y_masked = contents[mask]
        total = np.sum(y_masked)
        if total > 0 and len(c_masked) >= 3:
            cumsum = np.cumsum(y_masked) / total
            lo_q = (1.0 - core_fraction) / 2.0
            hi_q = 1.0 - lo_q
            lo = float(np.interp(lo_q, cumsum, c_masked))
            hi = float(np.interp(hi_q, cumsum, c_masked))
            # Narrow existing mask to the core percentile window
            mask = mask & (centers >= lo) & (centers <= hi)

    for iteration in range(n_iter):
        x = centers[mask]
        y = contents[mask]
        err = errors[mask]

        good = err > 0
        if np.sum(good) < 3:
            logger.debug("  iter %d: only %d bins with err>0, aborting", iteration, np.sum(good))
            return None

        x, y, err = x[good], y[good], err[good]

        # Use weighted moments for initial guess — more robust than argmax/range/4
        total = np.sum(y)
        if total > 0:
            mu0 = float(np.average(x, weights=y))
            var = np.average((x - mu0) ** 2, weights=y)
            sigma0 = float(np.sqrt(var)) if var > 0 else float((x[-1] - x[0]) / 4.0)
        else:
            mu0 = float(x[len(x) // 2])
            sigma0 = float((x[-1] - x[0]) / 4.0)
        A0 = float(np.max(y))

        try:
            popt, pcov = curve_fit(
                _gaussian, x, y,
                p0=[A0, mu0, sigma0],
                sigma=err,
                absolute_sigma=True,
                maxfev=10000,
            )
        except (RuntimeError, ValueError) as e:
            logger.debug("  iter %d: curve_fit failed: %s", iteration, e)
            return None

        mu = popt[1]
        sigma = abs(popt[2])
        diag = np.diag(pcov)
        # Guard against inf/nan covariance (ill-conditioned fit)
        perr = np.where(np.isfinite(diag) & (diag >= 0), np.sqrt(diag), np.abs(popt) * 0.1)

        # Refine window — floor at min_half_width so narrow peaks don't lose bins
        half_width = max(sigma_range * sigma, min_half_width)
        mask = (centers >= mu - half_width) & (centers <= mu + half_width)

    if np.sum(mask) < 3:
        logger.debug("Final window too narrow after iteration")
        return None

    return mu, sigma, perr[1], perr[2]


# ---------------------------------------------------------------------------
# Smearing term math (mirrors smearingPlots.py exactly)
# ---------------------------------------------------------------------------

def _smearing_term_absolute(sigma_data, sigma_mc, err_data=0., err_mc=0.):
    """sqrt(sigma_data^2 - sigma_mc^2) with error propagation."""
    if sigma_data < sigma_mc:
        logger.warning("DATA resolution (%.4g) < MC resolution (%.4g)", sigma_data, sigma_mc)
        return 0., 0.

    st = sqrt(sigma_data ** 2 - sigma_mc ** 2)
    if st < 1e-12:
        return 0., 0.

    dstdsd = sigma_data / st
    dstdsmc = -sigma_mc / st
    sigmast = sqrt(dstdsd ** 2 * err_data ** 2 + dstdsmc ** 2 * err_mc ** 2)
    if sigmast < 1e-12:
        sigmast = st * 0.01
    return st, sigmast


def _smearing_term_relative(sigma_data, mu_data, sigma_mc, mu_mc, err_data=0., err_mc=0.):
    """sqrt((sigma_data/mu_data)^2 - (sigma_mc/mu_mc)^2) with error propagation."""
    if mu_data == 0 or mu_mc == 0:
        return 0., 0.

    rel_data = sigma_data / mu_data
    rel_mc = sigma_mc / mu_mc

    if rel_data < rel_mc:
        logger.warning("DATA relative resolution (%.4g) < MC (%.4g)", rel_data, rel_mc)
        return 0., 0.

    st = sqrt(rel_data ** 2 - rel_mc ** 2)
    if st < 1e-6:
        return 0., 0.

    dstdsd = sigma_data / (mu_data ** 2 * st)
    dstdsmc = -sigma_mc / (mu_mc ** 2 * st)
    sigmast = sqrt(dstdsd ** 2 * err_data ** 2 + dstdsmc ** 2 * err_mc ** 2)
    if sigmast < 1e-6:
        sigmast = st * 0.01
    return st, sigmast


def _compute_smearing(smearing_cfg, mu_data, sigma_data, sigma_data_err,
                      mu_mc, sigma_mc, sigma_mc_err):
    if smearing_cfg.smearing_type == "relative":
        return _smearing_term_relative(sigma_data, mu_data, sigma_mc, mu_mc,
                                       sigma_data_err, sigma_mc_err)
    return _smearing_term_absolute(sigma_data, sigma_mc, sigma_data_err, sigma_mc_err)


# ---------------------------------------------------------------------------
# Shared: save a single fit diagnostic plot
# ---------------------------------------------------------------------------

def _normalise_to_unity(centers, contents, errors):
    """Normalise histogram to unit area (integral = 1). Returns (contents, errors)."""
    bw = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
    total = np.sum(contents) * bw
    if total <= 0:
        return contents.copy(), errors.copy()
    return contents / total, errors / total


def _effective_sigma_range(smearing_cfg):
    """Return the sigma_range actually used by the iterative fitter."""
    if smearing_cfg.core_fraction is not None:
        from scipy.special import erfinv as _erfinv
        return float(_erfinv(smearing_cfg.core_fraction) * np.sqrt(2))
    return smearing_cfg.sigma_range


def _draw_fit_on_ax(ax, x_label,
                    centers_d, contents_d, errors_d, fit_d, data_label, data_color,
                    centers_m, contents_m, errors_m, fit_m, mc_label, mc_color,
                    smearing_type, st, sigmast, title=None, compact=False, extras=None,
                    curve_n_sigma=None):
    """Draw data/MC histograms with Gaussian fits onto an existing axes.

    extras : list of (centers, contents, errors, fit_res, label, color), optional
        Additional samples to overlay (normalised and fitted, no smearing derived).
    """
    contents_d, errors_d = _normalise_to_unity(centers_d, contents_d, errors_d)
    contents_m, errors_m = _normalise_to_unity(centers_m, contents_m, errors_m)

    all_samples = [
        (centers_d, contents_d, errors_d, fit_d, data_label, data_color),
        (centers_m, contents_m, errors_m, fit_m, mc_label, mc_color),
    ]
    if extras:
        for (ec, ey, ee, ef, el, ecol) in extras:
            ey, ee = _normalise_to_unity(ec, ey, ee)
            all_samples.append((ec, ey, ee, ef, el, ecol))

    for centers, contents, errors, fit_res, label, color in all_samples:
        ax.errorbar(centers, contents, yerr=errors,
                    fmt="o", color=color, label=label, markersize=2 if compact else 3)

        if fit_res is not None:
            mu_f, sigma_f, _, _ = fit_res
            good = errors > 0
            if np.sum(good) >= 3:
                try:
                    popt_curve, _ = curve_fit(
                        _gaussian, centers[good], contents[good],
                        p0=[float(np.max(contents[good])), mu_f, sigma_f],
                        sigma=errors[good], absolute_sigma=True, maxfev=5000,
                    )
                    A_plot = popt_curve[0]
                except Exception:
                    A_plot = float(np.max(contents[good]))
            else:
                A_plot = float(np.max(contents))

            if curve_n_sigma is not None:
                curve_lo = max(centers[0],  mu_f - curve_n_sigma * sigma_f)
                curve_hi = min(centers[-1], mu_f + curve_n_sigma * sigma_f)
            else:
                curve_lo, curve_hi = centers[0], centers[-1]
            x_curve = np.linspace(curve_lo, curve_hi, 300)
            ax.plot(x_curve, _gaussian(x_curve, A_plot, mu_f, sigma_f),
                    color=color, linewidth=1.5, linestyle="--")

    fs = 7 if compact else 9
    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel("A.U.", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs - 1)
    if title:
        ax.set_title(title, fontsize=fs)

    smearing_label = "Rel." if smearing_type == "relative" else "Abs."
    if fit_d is not None and fit_m is not None:
        mu_d, sigma_d, _, sigma_d_err = fit_d
        mu_m, sigma_m, _, sigma_m_err = fit_m
        ann = (f"D: $\\mu={mu_d:.3g}$, $\\sigma={sigma_d:.3g}$\n"
               f"MC: $\\mu={mu_m:.3g}$, $\\sigma={sigma_m:.3g}$\n"
               f"{smearing_label}: ${st:.3g}\\pm{sigmast:.1g}$")
        ax.text(0.97, 0.97, ann, transform=ax.transAxes, fontsize=fs - 1,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


def _save_fit_plot(outpath, x_label,
                   centers_d, contents_d, errors_d, fit_d, data_label, data_color,
                   centers_m, contents_m, errors_m, fit_m, mc_label, mc_color,
                   smearing_type, st, sigmast, title=None, output_format="pdf", extras=None,
                   curve_n_sigma=None):
    """Save a standalone plot showing data/MC histograms with fitted Gaussians."""
    fig, ax = plt.subplots(figsize=(8, 5))
    add_hps_label(ax)
    _draw_fit_on_ax(ax, x_label,
                    centers_d, contents_d, errors_d, fit_d, data_label, data_color,
                    centers_m, contents_m, errors_m, fit_m, mc_label, mc_color,
                    smearing_type, st, sigmast, title=title, compact=False, extras=extras,
                    curve_n_sigma=curve_n_sigma)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outpath)


# ---------------------------------------------------------------------------
# 1D smearing
# ---------------------------------------------------------------------------

def _plot_smearing_1d(plot_cfg, hist_cfg, region_cfg, results, samples_map, output_dir,
                      output_format):
    smearing_cfg = plot_cfg.smearing
    region_name = region_cfg.name
    hist_name = hist_cfg.name

    region_results = results.get(region_name, {})
    hdata_data = region_results.get(smearing_cfg.data_sample, {}).get(hist_name)
    hdata_mc = region_results.get(smearing_cfg.mc_sample, {}).get(hist_name)

    if hdata_data is None or hdata_mc is None:
        logger.warning("Missing histograms for smearing 1D: %s / %s", region_name, hist_name)
        return None

    fit_range = smearing_cfg.fit_range if smearing_cfg.fit_range else None

    logger.debug("1D fit %s/%s: %d data bins, %d MC bins, fit_range=%s",
                 region_name, hist_name,
                 np.sum(hdata_data.bin_contents > 0),
                 np.sum(hdata_mc.bin_contents > 0), fit_range)

    result_data = _iterative_gauss_fit(
        hdata_data.bin_centers, hdata_data.bin_contents, hdata_data.bin_errors,
        sigma_range=smearing_cfg.sigma_range, fit_range=fit_range,
        core_fraction=smearing_cfg.core_fraction,
    )
    result_mc = _iterative_gauss_fit(
        hdata_mc.bin_centers, hdata_mc.bin_contents, hdata_mc.bin_errors,
        sigma_range=smearing_cfg.sigma_range, fit_range=fit_range,
        core_fraction=smearing_cfg.core_fraction,
    )

    if result_data is None or result_mc is None:
        logger.warning("Fit failed for smearing 1D: %s / %s (data=%s, mc=%s)",
                       region_name, hist_name,
                       "OK" if result_data else "FAIL",
                       "OK" if result_mc else "FAIL")
        return None

    mu_data, sigma_data, mu_err_data, sigma_err_data = result_data
    mu_mc, sigma_mc, mu_err_mc, sigma_err_mc = result_mc

    st, sigmast = _compute_smearing(smearing_cfg, mu_data, sigma_data, sigma_err_data,
                                    mu_mc, sigma_mc, sigma_err_mc)

    logger.info("%s / %s: sigma_data=%.4g±%.4g, sigma_mc=%.4g±%.4g, smearing=%.4g±%.4g",
                region_name, hist_name, sigma_data, sigma_err_data,
                sigma_mc, sigma_err_mc, st, sigmast)

    # Fit extra samples (for overlay, no smearing derived from them)
    extras = []
    for extra_name in smearing_cfg.extra_samples:
        h_extra = region_results.get(extra_name, {}).get(hist_name)
        if h_extra is None:
            logger.warning("Extra sample '%s' missing for %s/%s", extra_name, region_name, hist_name)
            continue
        fit_extra = _iterative_gauss_fit(
            h_extra.bin_centers, h_extra.bin_contents, h_extra.bin_errors,
            sigma_range=smearing_cfg.sigma_range, fit_range=fit_range,
            core_fraction=smearing_cfg.core_fraction,
        )
        extras.append((
            h_extra.bin_centers, h_extra.bin_contents, h_extra.bin_errors,
            fit_extra,
            samples_map[extra_name].label,
            samples_map[extra_name].color,
        ))

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{plot_cfg.name}_{region_name}_{hist_name}.{output_format}"

    _save_fit_plot(
        outpath=outdir / fname,
        x_label=hist_cfg.x_label or hist_cfg.variable,
        centers_d=hdata_data.bin_centers, contents_d=hdata_data.bin_contents,
        errors_d=hdata_data.bin_errors, fit_d=result_data,
        data_label=samples_map[smearing_cfg.data_sample].label,
        data_color=samples_map[smearing_cfg.data_sample].color,
        centers_m=hdata_mc.bin_centers, contents_m=hdata_mc.bin_contents,
        errors_m=hdata_mc.bin_errors, fit_m=result_mc,
        mc_label=samples_map[smearing_cfg.mc_sample].label,
        mc_color=samples_map[smearing_cfg.mc_sample].color,
        smearing_type=smearing_cfg.smearing_type,
        st=st, sigmast=sigmast,
        title=f"{region_name} / {hist_name}",
        output_format=output_format,
        extras=extras or None,
        curve_n_sigma=_effective_sigma_range(smearing_cfg),
    )

    return {"mu_data": mu_data, "sigma_data": sigma_data, "sigma_data_err": sigma_err_data,
            "mu_mc": mu_mc, "sigma_mc": sigma_mc, "sigma_mc_err": sigma_err_mc,
            "smearing": st, "smearing_err": sigmast}


# ---------------------------------------------------------------------------
# 2D smearing
# ---------------------------------------------------------------------------

def _plot_smearing_2d(plot_cfg, hist_cfg, region_cfg, results, samples_map, output_dir,
                      output_format, beam_energy=None):
    smearing_cfg = plot_cfg.smearing
    region_name = region_cfg.name
    hist_name = hist_cfg.name

    region_results = results.get(region_name, {})
    h2d_data = region_results.get(smearing_cfg.data_sample, {}).get(hist_name)
    h2d_mc = region_results.get(smearing_cfg.mc_sample, {}).get(hist_name)

    if not isinstance(h2d_data, Histogram2DData) or not isinstance(h2d_mc, Histogram2DData):
        logger.warning("Expected Histogram2DData for smearing 2D: %s / %s", region_name, hist_name)
        return None

    fit_range = smearing_cfg.fit_range if smearing_cfg.fit_range else None
    x_centers = h2d_data.x_centers
    x_edges = h2d_data.x_edges
    y_centers = h2d_data.y_centers
    n_x = len(x_centers)

    sigma_data_arr = np.zeros(n_x)
    sigma_mc_arr = np.zeros(n_x)
    sigma_data_err_arr = np.zeros(n_x)
    sigma_mc_err_arr = np.zeros(n_x)
    mu_data_arr = np.zeros(n_x)
    mu_mc_arr = np.zeros(n_x)
    mu_data_err_arr = np.zeros(n_x)
    mu_mc_err_arr = np.zeros(n_x)
    smearing_arr = np.zeros(n_x)
    smearing_err_arr = np.zeros(n_x)
    valid = np.zeros(n_x, dtype=bool)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    x_label = hist_cfg.x_label or hist_cfg.variable
    y_label = hist_cfg.y_label_2d or hist_cfg.y_variable
    data_label = samples_map[smearing_cfg.data_sample].label
    mc_label = samples_map[smearing_cfg.mc_sample].label
    data_color = samples_map[smearing_cfg.data_sample].color
    mc_color = samples_map[smearing_cfg.mc_sample].color

    # Pre-load extra sample 2D histograms
    extra_h2ds = []
    for extra_name in smearing_cfg.extra_samples:
        h2d_extra = region_results.get(extra_name, {}).get(hist_name)
        if not isinstance(h2d_extra, Histogram2DData):
            logger.warning("Extra sample '%s' missing 2D histogram for %s/%s",
                           extra_name, region_name, hist_name)
            continue
        extra_h2ds.append((h2d_extra, samples_map[extra_name].label,
                           samples_map[extra_name].color, extra_name,
                           samples_map[extra_name].derive_smearing))

    # Per-extra-sample summary arrays (sigma vs x, smearing vs x, mu vs x)
    n_extras = len(extra_h2ds)
    extra_sigma_arrs     = [np.zeros(n_x) for _ in range(n_extras)]
    extra_sigma_err_arrs = [np.zeros(n_x) for _ in range(n_extras)]
    extra_smearing_arrs  = [np.zeros(n_x) for _ in range(n_extras)]
    extra_smearing_err_arrs = [np.zeros(n_x) for _ in range(n_extras)]
    extra_mu_arrs        = [np.zeros(n_x) for _ in range(n_extras)]
    extra_mu_err_arrs    = [np.zeros(n_x) for _ in range(n_extras)]
    extra_valid          = [np.zeros(n_x, dtype=bool) for _ in range(n_extras)]

    # Collect per-bin data for the combined canvas
    bin_plot_data = []

    for ix in range(n_x):
        y_data = h2d_data.contents[ix, :]
        e_data = h2d_data.errors[ix, :]
        y_mc = h2d_mc.contents[ix, :]
        e_mc = h2d_mc.errors[ix, :]

        x_lo = x_edges[ix]
        x_hi = x_edges[ix + 1]
        bin_title = f"{x_label}: [{x_lo:.3g}, {x_hi:.3g}]"

        logger.debug("2D fit %s/%s bin %d [%.3g, %.3g]: data=%d MC=%d non-zero bins",
                     region_name, hist_name, ix, x_lo, x_hi,
                     np.sum(y_data > 0), np.sum(y_mc > 0))

        r_data = _iterative_gauss_fit(y_centers, y_data, e_data,
                                      sigma_range=smearing_cfg.sigma_range,
                                      fit_range=fit_range,
                                      core_fraction=smearing_cfg.core_fraction)
        r_mc = _iterative_gauss_fit(y_centers, y_mc, e_mc,
                                    sigma_range=smearing_cfg.sigma_range,
                                    fit_range=fit_range,
                                    core_fraction=smearing_cfg.core_fraction)

        if r_data is None or r_mc is None:
            logger.warning("Fit failed in x-bin %d [%.3g, %.3g] (%s/%s): data=%s mc=%s",
                           ix, x_lo, x_hi, region_name, hist_name,
                           "OK" if r_data else "FAIL", "OK" if r_mc else "FAIL")
        else:
            mu_d, sigma_d, mu_d_err, sigma_d_err = r_data
            mu_m, sigma_m, mu_m_err, sigma_m_err = r_mc
            st, sigmast = _compute_smearing(smearing_cfg, mu_d, sigma_d, sigma_d_err,
                                            mu_m, sigma_m, sigma_m_err)
            mu_data_arr[ix] = mu_d
            mu_mc_arr[ix] = mu_m
            mu_data_err_arr[ix] = mu_d_err
            mu_mc_err_arr[ix] = mu_m_err
            sigma_data_arr[ix] = sigma_d
            sigma_mc_arr[ix] = sigma_m
            sigma_data_err_arr[ix] = sigma_d_err
            sigma_mc_err_arr[ix] = sigma_m_err
            smearing_arr[ix] = st
            smearing_err_arr[ix] = sigmast
            valid[ix] = True

        # Fit extra samples for this bin
        bin_extras = []
        for ei, (h2d_ex, ex_label, ex_color, _, ex_derive) in enumerate(extra_h2ds):
            y_ex = h2d_ex.contents[ix, :]
            e_ex = h2d_ex.errors[ix, :]
            r_ex = _iterative_gauss_fit(y_centers, y_ex, e_ex,
                                        sigma_range=smearing_cfg.sigma_range,
                                        fit_range=fit_range,
                                        core_fraction=smearing_cfg.core_fraction)
            bin_extras.append((y_centers, y_ex, e_ex, r_ex, ex_label, ex_color))

            # Accumulate sigma, mu, and (optionally) derived smearing for summary plot
            if r_ex is not None:
                mu_ex, sigma_ex, mu_ex_err, sigma_ex_err = r_ex
                extra_sigma_arrs[ei][ix]     = sigma_ex
                extra_sigma_err_arrs[ei][ix] = sigma_ex_err
                extra_mu_arrs[ei][ix]        = mu_ex
                extra_mu_err_arrs[ei][ix]    = mu_ex_err
                extra_valid[ei][ix]          = True
                if ex_derive and valid[ix]:
                    st_ex, sigmast_ex = _compute_smearing(
                        smearing_cfg,
                        mu_data_arr[ix], sigma_data_arr[ix], sigma_data_err_arr[ix],
                        mu_ex, sigma_ex, sigma_ex_err,
                    )
                    extra_smearing_arrs[ei][ix]     = st_ex
                    extra_smearing_err_arrs[ei][ix] = sigmast_ex

        bin_plot_data.append(dict(
            y_data=y_data, e_data=e_data, r_data=r_data,
            y_mc=y_mc, e_mc=e_mc, r_mc=r_mc,
            st=smearing_arr[ix] if valid[ix] else 0.,
            sigmast=smearing_err_arr[ix] if valid[ix] else 0.,
            title=bin_title,
            extras=bin_extras,
        ))

    # --- Combined fit-diagnostic canvas (all bins on one figure) ---
    n_cols = min(4, n_x)
    n_rows = (n_x + n_cols - 1) // n_cols
    fig_fits, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4.5 * n_cols, 3.5 * n_rows),
                                  squeeze=False)
    fig_fits.suptitle(f"{plot_cfg.name}  |  {region_name}  |  {hist_name}",
                      fontsize=10, y=1.01)

    for ix, bpd in enumerate(bin_plot_data):
        row, col = divmod(ix, n_cols)
        ax = axes[row][col]
        _draw_fit_on_ax(
            ax, y_label,
            y_centers, bpd["y_data"], bpd["e_data"], bpd["r_data"], data_label, data_color,
            y_centers, bpd["y_mc"],   bpd["e_mc"],   bpd["r_mc"],   mc_label,   mc_color,
            smearing_cfg.smearing_type, bpd["st"], bpd["sigmast"],
            title=bpd["title"], compact=True, extras=bpd["extras"] or None,
            curve_n_sigma=_effective_sigma_range(smearing_cfg),
        )

    # Hide any unused axes in the last row
    for ix in range(n_x, n_rows * n_cols):
        row, col = divmod(ix, n_cols)
        axes[row][col].set_visible(False)

    fits_fname = f"{plot_cfg.name}_{region_name}_{hist_name}_fits.{output_format}"
    fig_fits.savefig(outdir / fits_fname, bbox_inches="tight")
    plt.close(fig_fits)
    logger.info("Saved fit canvas: %s", outdir / fits_fname)

    def _pad_yaxis(ax, pad=0.25):
        """Expand y-axis limits by *pad* fraction beyond the current data range."""
        lo, hi = ax.get_ylim()
        span = hi - lo
        if span > 0:
            ax.set_ylim(lo - pad * span, hi + pad * span)

    # --- Summary plot: sigma, smearing, and scale correction vs X ---
    smearing_label = "Relative smearing" if smearing_cfg.smearing_type == "relative" else "Absolute smearing"
    mc_derive = samples_map[smearing_cfg.mc_sample].derive_smearing
    show_smearing_panel = mc_derive or any(ex_derive for (_, _, _, _, ex_derive) in extra_h2ds)

    # Scale correction panel: shown when scale_panel=True and either
    #   - scale_reference is None  →  ratio mode (needs beam_energy)
    #   - scale_reference is a float  →  absolute mode (beam_energy not required)
    _scale_ratio_mode = smearing_cfg.scale_reference is None
    show_scale_panel = (
        smearing_cfg.scale_panel
        and np.any(valid)
        and (beam_energy is not None if _scale_ratio_mode else True)
    )

    # Build panel layout dynamically
    height_ratios = [1.4]
    if show_smearing_panel:
        height_ratios.append(0.6)
    if show_scale_panel:
        height_ratios.append(0.4)

    n_panels = len(height_ratios)
    fig_height = 4.5 + 1.5 * n_panels
    if n_panels == 1:
        fig, ax_top = plt.subplots(1, 1, figsize=(8, fig_height))
        ax_bot = None
        ax_scale = None
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, fig_height), sharex=True,
                                 gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05})
        ax_top = axes[0]
        panel_idx = 1
        if show_smearing_panel:
            ax_bot = axes[panel_idx]
            panel_idx += 1
        else:
            ax_bot = None
        if show_scale_panel:
            ax_scale = axes[panel_idx]
        else:
            ax_scale = None

    xv = x_centers[valid]
    ax_top.errorbar(xv, sigma_data_arr[valid], yerr=sigma_data_err_arr[valid],
                    fmt="o", color=data_color, label=f"$\\sigma$ {data_label}", markersize=4)
    ax_top.errorbar(xv, sigma_mc_arr[valid], yerr=sigma_mc_err_arr[valid],
                    fmt="s", color=mc_color, label=f"$\\sigma$ {mc_label}", markersize=4)
    for ei, (_, ex_label, ex_color, _, _) in enumerate(extra_h2ds):
        ev = extra_valid[ei]
        if np.any(ev):
            ax_top.errorbar(x_centers[ev], extra_sigma_arrs[ei][ev],
                            yerr=extra_sigma_err_arrs[ei][ev],
                            fmt="^", color=ex_color, label=f"$\\sigma$ {ex_label}", markersize=4)
    ax_top.set_ylabel(f"$\\sigma$ ({y_label})")
    if ax_bot is None and ax_scale is None:
        ax_top.set_xlabel(x_label)
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=18)
    add_hps_label(ax_top)
    _pad_yaxis(ax_top, pad=0.25)

    if ax_bot is not None:
        if mc_derive:
            ax_bot.errorbar(xv, smearing_arr[valid], yerr=smearing_err_arr[valid],
                            fmt="o", color=mc_color, label=mc_label, markersize=4)
        for ei, (_, ex_label, ex_color, _, ex_derive) in enumerate(extra_h2ds):
            if not ex_derive:
                continue
            ev = extra_valid[ei] & valid
            if np.any(ev):
                ax_bot.errorbar(x_centers[ev], extra_smearing_arrs[ei][ev],
                                yerr=extra_smearing_err_arrs[ei][ev],
                                fmt="^", color=ex_color, label=ex_label, markersize=4)
        ax_bot.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        if ax_scale is None:
            ax_bot.set_xlabel(x_label)
        ax_bot.set_ylabel(smearing_label)
        _pad_yaxis(ax_bot, pad=0.25)

    if ax_scale is not None:
        xv = x_centers[valid]

        if _scale_ratio_mode:
            # Ratio mode: show μ / beam_energy, reference line at 1
            def _sv(mu, err):
                return mu / beam_energy, err / beam_energy
            ref_line = 1.0
            scale_ylabel = "$\\mu / E_{\\rm beam}$"
        else:
            # Absolute mode: show μ directly, reference line at scale_reference
            def _sv(mu, err):
                return mu.copy(), err.copy()
            ref_line = float(smearing_cfg.scale_reference)
            scale_ylabel = "$\\mu$"

        # Data sample (always shown)
        yv, ye = _sv(mu_data_arr[valid], mu_data_err_arr[valid])
        ax_scale.errorbar(xv, yv, yerr=ye,
                          fmt="o", color=data_color, label=data_label, markersize=4)

        # MC sample if flagged
        if samples_map[smearing_cfg.mc_sample].show_scale:
            yv, ye = _sv(mu_mc_arr[valid], mu_mc_err_arr[valid])
            ax_scale.errorbar(xv, yv, yerr=ye,
                              fmt="s", color=mc_color, label=mc_label, markersize=4)

        # Extra samples if flagged
        for ei, (_, ex_label, ex_color, ex_name, _) in enumerate(extra_h2ds):
            if not samples_map[ex_name].show_scale:
                continue
            ev = extra_valid[ei]
            if np.any(ev):
                yv, ye = _sv(extra_mu_arrs[ei][ev], extra_mu_err_arrs[ei][ev])
                ax_scale.errorbar(x_centers[ev], yv, yerr=ye,
                                  fmt="^", color=ex_color, label=ex_label, markersize=4)

        ax_scale.axhline(ref_line, color="gray", linestyle="--", linewidth=0.8)
        ax_scale.set_xlabel(x_label)
        ax_scale.set_ylabel(scale_ylabel)
        _pad_yaxis(ax_scale, pad=0.25)

    fname = f"{plot_cfg.name}_{region_name}_{hist_name}.{output_format}"
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", outdir / fname)

    return {
        "x_centers": x_centers[valid].tolist(),
        "sigma_data": sigma_data_arr[valid].tolist(),
        "sigma_mc": sigma_mc_arr[valid].tolist(),
        "smearing": smearing_arr[valid].tolist(),
        "smearing_err": smearing_err_arr[valid].tolist(),
        # Full arrays aligned with bin edges — used by build_tool_json
        "x_edges": x_edges.tolist(),
        "smearing_all": smearing_arr.tolist(),   # length n_x; 0 for failed fits
        "mu_data_all": mu_data_arr.tolist(),     # length n_x; 0 for failed fits
        "mu_mc_all": mu_mc_arr.tolist(),         # length n_x; 0 for failed fits
    }


# ---------------------------------------------------------------------------
# Tool JSON builder
# ---------------------------------------------------------------------------

def build_tool_json(smearing_cfg, smearing_results, hist_map):
    """Build a TrackSmearingTool-compatible JSON dict from collected smearing results.

    The returned dict can be merged into an existing tool JSON file.
    Returns None if tool_section, top_region, or bot_region are not configured.
    """
    if not smearing_cfg.tool_section or not smearing_cfg.top_region or not smearing_cfg.bot_region:
        return None

    section = smearing_cfg.tool_section
    top_region = smearing_cfg.top_region
    bot_region = smearing_cfg.bot_region

    tool = {}

    # Set smearing-type flags based on section
    if section == "pSmearing":
        tool["relSmearingP"] = (smearing_cfg.smearing_type == "relative")
    elif section == "omegaSmearing":
        tool["smearOmega"] = True

    for key, result in smearing_results.items():
        parts = key.split("/", 1)
        if len(parts) != 2:
            continue
        region_name, hist_name = parts
        hist_cfg = hist_map.get(hist_name)
        if hist_cfg is None:
            continue

        is_top = (region_name == top_region)
        is_bot = (region_name == bot_region)
        if not is_top and not is_bot:
            continue

        vol = "top" if is_top else "bot"

        if "x_edges" in result:
            # 2D result — binned lookup table; key encodes the lookup variable
            tool_var = hist_cfg.tool_variable_name
            if not tool_var:
                continue
            binned_key = f"{section}_binned_{tool_var}"
            if binned_key not in tool:
                tool[binned_key] = {}
            tool[binned_key][vol] = {
                "bin_edges": result["x_edges"],
                "values": result["smearing_all"],
                "mu_data": result.get("mu_data_all", []),
                "mu_mc":   result.get("mu_mc_all", []),
            }
            logger.debug("Tool JSON: %s[%s] binned (%d bins, var=%s)",
                         section, vol, len(result["smearing_all"]), tool_var)
        else:
            # 1D result — scalar smearing + means
            if section not in tool:
                tool[section] = {}
            tool[section][vol] = result.get("smearing", 0.0)
            # Store means in a sub-dict so the scalar format stays backward-compatible
            if "mu_data" not in tool[section]:
                tool[section]["mu_data"] = {}
                tool[section]["mu_mc"] = {}
            tool[section]["mu_data"][vol] = result.get("mu_data", 0.0)
            tool[section]["mu_mc"][vol]   = result.get("mu_mc", 0.0)
            logger.debug("Tool JSON: %s[%s] = %.4g (mu_data=%.4g, mu_mc=%.4g)",
                         section, vol, tool[section][vol],
                         tool[section]["mu_data"][vol], tool[section]["mu_mc"][vol])

    return tool


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plot_smearing(plot_cfg, hist_cfg, region_cfg, results, samples_map, output_dir,
                  output_format="pdf", beam_energy=None):
    """Derive and plot smearing factors for a given histogram.

    Dispatches to 1D or 2D based on whether hist_cfg.y_variable is set.
    Returns a dict of smearing results for JSON output, or None on failure.
    """
    if plot_cfg.smearing is None:
        logger.warning("plot_type=smearing but no 'smearing:' config block for plot '%s'",
                       plot_cfg.name)
        return None

    if hist_cfg.y_variable:
        return _plot_smearing_2d(plot_cfg, hist_cfg, region_cfg, results,
                                 samples_map, output_dir, output_format,
                                 beam_energy=beam_energy)
    return _plot_smearing_1d(plot_cfg, hist_cfg, region_cfg, results,
                             samples_map, output_dir, output_format)
