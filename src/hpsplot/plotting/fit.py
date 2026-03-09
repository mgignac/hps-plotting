"""Histogram fitting utilities."""

import logging

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def _exponential(x, A, lam):
    return A * np.exp(-x / lam)


_MODELS = {
    "exponential": (_exponential, ["A", r"\lambda"]),
}


def fit_histogram(hdata, fit_cfg, hist_cfg):
    """Fit a HistogramData with the configured function.

    Returns (popt, pcov, x_curve, y_curve, param_labels) or None on failure.
    """
    model_func, param_labels = _MODELS[fit_cfg.function]

    centers = hdata.bin_centers
    contents = hdata.bin_contents
    errors = hdata.bin_errors

    # Determine fit range
    x_min = fit_cfg.x_min if fit_cfg.x_min is not None else hist_cfg.x_min
    x_max = fit_cfg.x_max if fit_cfg.x_max is not None else hist_cfg.x_max

    mask = (centers >= x_min) & (centers <= x_max) & (errors > 0)
    if np.sum(mask) < len(param_labels):
        logger.warning("Not enough bins for fit (need >= %d, got %d)",
                       len(param_labels), np.sum(mask))
        return None

    x_fit = centers[mask]
    y_fit = contents[mask]
    sigma = errors[mask]

    p0 = fit_cfg.p0 if fit_cfg.p0 else None

    try:
        popt, pcov = curve_fit(model_func, x_fit, y_fit, p0=p0, sigma=sigma,
                               absolute_sigma=True)
    except RuntimeError as e:
        logger.warning("Fit failed: %s", e)
        return None

    x_curve = np.linspace(x_min, x_max, 100)
    y_curve = model_func(x_curve, *popt)

    return popt, pcov, x_curve, y_curve, param_labels
