"""Processing orchestration: samples x regions x histograms."""

import logging

import numpy as np

from .config import Config
from .histogram import fill_histogram
from .region import Region
from .sample import Sample, compute_luminosity
from .utils import safe_evaluate

logger = logging.getLogger(__name__)


def process(config: Config):
    """Run the processing loop over all samples, regions, and histograms.

    Parameters
    ----------
    config : Config
        Fully loaded configuration.

    Returns
    -------
    dict
        results[region_name][sample_name][hist_name] → HistogramData
    """
    # Build lookup maps
    sample_map = {s.name: s for s in config.samples}
    region_map = {r.name: r for r in config.regions}
    hist_map = {h.name: h for h in config.histograms}

    # Compute luminosity from data files + lumi file if configured
    if config.lumi_file:
        data_sample = next(
            (s for s in config.samples if s.sample_type == "data"), None
        )
        if data_sample:
            config.luminosity = compute_luminosity(
                data_sample.directory, config.lumi_file
            )
        else:
            logger.warning("lumi_file set but no data sample found — using luminosity=%.4g",
                           config.luminosity)

    # Determine which samples, regions, histograms are actually used by plots
    needed_samples = set()
    needed_regions = set()
    needed_hists = set()

    for plot in config.plots:
        needed_samples |= set(plot.samples)
        needed_regions |= set(plot.regions)
        needed_hists |= set(plot.histograms)

    # If no plots defined, process everything
    if not config.plots:
        needed_samples = set(sample_map.keys())
        needed_regions = set(region_map.keys())
        needed_hists = set(hist_map.keys())

    # Build objects
    samples = {}
    for name in needed_samples:
        if name not in sample_map:
            raise ValueError(f"Sample '{name}' referenced in plot but not defined.")
        samples[name] = Sample(sample_map[name])

    regions = {}
    for name in needed_regions:
        if name not in region_map:
            raise ValueError(f"Region '{name}' referenced in plot but not defined.")
        regions[name] = Region(region_map[name])

    hists = {}
    for name in needed_hists:
        if name not in hist_map:
            raise ValueError(f"Histogram '{name}' referenced in plot but not defined.")
        hists[name] = hist_map[name]

    # Process each sample
    results = {}
    for sample_name, sample in samples.items():
        # Collect all needed branches
        hist_list = list(hists.values())
        region_list = list(regions.values())
        region_configs = [r.config for r in region_list]
        branches = sample.get_needed_branches(hist_list, region_configs)

        # Load data
        data = sample.load(branches, aliases=config.aliases)

        # Compute luminosity weight if enabled
        lumi_weight = 1.0
        if sample.config.lumi_scale:
            lumi_weight, xsec, n_gen = sample.get_lumi_weight(config.luminosity)
            total_scale = sample.scale * lumi_weight
            logger.info(
                "Lumi weight for '%s':\n"
                "    cross_section  = %.4g\n"
                "    luminosity     = %.4g\n"
                "    n_generated    = %d\n"
                "    lumi_weight    = %.4g\n"
                "    sample scale   = %.4g\n"
                "    total scale    = %.4g",
                sample_name, xsec, config.luminosity, n_gen,
                lumi_weight, sample.scale, total_scale,
            )

        # Process each region
        for region_name, region in regions.items():
            if region_name not in results:
                results[region_name] = {}

            # Compute selection mask
            mask = region.apply(data)

            # Process each histogram
            results[region_name][sample_name] = {}
            for hist_name, hist_cfg in hists.items():
                # Evaluate variable expression on masked data
                values = safe_evaluate(hist_cfg.variable, data, mask=mask)

                # Evaluate weight expression on masked data
                weights = safe_evaluate(sample.weight_expr, data, mask=mask)

                # Apply scale factor and luminosity weight
                total_scale = sample.scale * lumi_weight
                if total_scale != 1.0:
                    weights = np.asarray(weights, dtype=float) * total_scale

                # Handle scalar weight (e.g., "1.0")
                if np.ndim(weights) == 0:
                    weights = np.full_like(values, float(weights), dtype=float)

                # Fill histogram
                hdata = fill_histogram(
                    values, weights,
                    hist_cfg.bins, hist_cfg.x_min, hist_cfg.x_max,
                )
                results[region_name][sample_name][hist_name] = hdata

                logger.info(
                    "  %s / %s / %s: integral = %.1f",
                    sample_name, region_name, hist_name, hdata.integral,
                )

    return results
