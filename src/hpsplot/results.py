"""Processing orchestration: samples x regions x histograms."""

import copy
import logging

import numpy as np

from .config import Config, SampleConfig
from .histogram import fill_histogram, fill_histogram_2d
from .region import Region
from .sample import Sample, compute_luminosity
from .signal_scaling import compute_eq4_scale, count_data_in_window
from .utils import safe_evaluate

logger = logging.getLogger(__name__)


def _process_single_sample(sample, lumi_weight, regions, hists, config):
    """Process one sample (single directory) through all regions and histograms.

    Returns
    -------
    dict
        sub_results[region_name][hist_name] → HistogramData
    """
    hist_list = list(hists.values())
    region_list = list(regions.values())
    region_configs = [r.config for r in region_list]
    branches = sample.get_needed_branches(hist_list, region_configs)

    # Merge global aliases with per-sample overrides (sample wins on conflicts)
    effective_aliases = {**config.aliases, **sample.config.aliases}
    data = sample.load(branches, aliases=effective_aliases)

    # Sample-level selection mask
    sample_mask = None
    if sample.config.selection:
        sample_mask = np.asarray(
            safe_evaluate(sample.config.selection, data), dtype=bool
        )
        n_pass = np.sum(sample_mask)
        n_total = len(sample_mask)
        logger.info(
            "  Sample selection for '%s': %d / %d pass (%.1f%%)",
            sample.name, n_pass, n_total,
            100 * n_pass / n_total if n_total > 0 else 0,
        )

    sub_results = {}
    for region_name, region in regions.items():
        mask = region.apply(data)
        if sample_mask is not None:
            mask = mask & sample_mask

        sub_results[region_name] = {}
        for hist_name, hist_cfg in hists.items():
            values = safe_evaluate(hist_cfg.variable, data, mask=mask)
            weights = safe_evaluate(sample.weight_expr, data, mask=mask)

            total_scale = sample.scale * lumi_weight
            if total_scale != 1.0:
                weights = np.asarray(weights, dtype=float) * total_scale

            if np.ndim(weights) == 0:
                weights = np.full_like(values, float(weights), dtype=float)

            if hist_cfg.y_variable:
                y_values = safe_evaluate(hist_cfg.y_variable, data, mask=mask)
                hdata = fill_histogram_2d(
                    values, y_values, weights,
                    hist_cfg.bins, hist_cfg.x_min, hist_cfg.x_max,
                    hist_cfg.y_bins, hist_cfg.y_min, hist_cfg.y_max,
                )
                logger.info(
                    "  %s / %s / %s: total weight = %.1f",
                    sample.name, region_name, hist_name, np.sum(hdata.contents),
                )
            else:
                hdata = fill_histogram(
                    values, weights,
                    hist_cfg.bins, hist_cfg.x_min, hist_cfg.x_max,
                )
                logger.info(
                    "  %s / %s / %s: integral = %.1f",
                    sample.name, region_name, hist_name, hdata.integral,
                )
            sub_results[region_name][hist_name] = hdata

    return sub_results


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
    # Nothing to do if there are no histograms to fill
    if not config.histograms:
        return {}

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
                data_sample.directories, config.lumi_file
            )
        else:
            logger.warning("lumi_file set but no data sample found — using luminosity=%.4g",
                           config.luminosity)

    # Apply Eq. 4 data-driven signal scaling if a scaling region is defined
    scaling_region_cfg = next(
        (r for r in config.regions if r.is_scaling_region), None
    )
    if scaling_region_cfg is not None:
        data_cfg = next(
            (s for s in config.samples if s.sample_type == "data"), None
        )
        mass_var = config.scaling_mass_variable
        mass_hw = config.scaling_mass_window

        if data_cfg is None:
            logger.warning(
                "Scaling region '%s' defined but no data sample found — skipping Eq. 4 scaling.",
                scaling_region_cfg.name,
            )
        elif not mass_var:
            logger.warning(
                "Scaling region '%s' defined but scaling_mass_variable not set — skipping Eq. 4 scaling.",
                scaling_region_cfg.name,
            )
        else:
            # Count data once per unique ap_mass value to avoid reloading.
            data_counts_cache = {}
            for s in config.samples:
                if s.sample_type == "signal" and s.ap_mass is not None:
                    if s.ap_mass not in data_counts_cache:
                        data_counts_cache[s.ap_mass] = count_data_in_window(
                            data_cfg, scaling_region_cfg,
                            mass_var, mass_hw, s.ap_mass, config.aliases,
                        )
                    scale = compute_eq4_scale(
                        s, data_counts_cache[s.ap_mass],
                        scaling_region_cfg, config.aliases,
                        mass_hw, config.scaling_rad_frac,
                    )
                    logger.info(
                        "Applying Eq. 4 scale %.4g to signal sample '%s' "
                        "(was %.4g, now %.4g)",
                        scale, s.name, s.scale, s.scale * scale,
                    )
                    s.scale *= scale

    # Determine which samples, regions, histograms are actually used by plots
    needed_samples = set()
    needed_regions = set()
    needed_hists = set()

    for plot in config.plots:
        if plot.plot_type == "abcd":
            continue  # ABCD loads its own data; skip histogram pipeline
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
        needs_split = (
            sample.config.lumi_scale
            and len(sample.config.directories) > 1
        )

        if needs_split:
            # Process each directory as a sub-sample with its own lumi weight,
            # then merge the histograms.
            logger.info("Processing sample '%s' (%d directories, per-directory lumi scaling)",
                        sample_name, len(sample.config.directories))
            merged = None

            for i, directory in enumerate(sample.config.directories):
                # Create a single-directory config copy
                sub_cfg = copy.copy(sample.config)
                sub_cfg.directory = directory
                sub_cfg.directories = [directory]
                sub_sample = Sample(sub_cfg)

                lumi_weight, xsec, n_gen = sub_sample.get_lumi_weight(config.luminosity)
                total_scale = sub_sample.scale * lumi_weight
                logger.info(
                    "Lumi weight for '%s' [dir %d/%d]:\n"
                    "    directory      = %s\n"
                    "    cross_section  = %.4g\n"
                    "    luminosity     = %.4g\n"
                    "    n_generated    = %d\n"
                    "    lumi_weight    = %.4g\n"
                    "    sample scale   = %.4g\n"
                    "    total scale    = %.4g",
                    sample_name, i + 1, len(sample.config.directories),
                    directory, xsec, config.luminosity, n_gen,
                    lumi_weight, sub_sample.scale, total_scale,
                )

                sub_results = _process_single_sample(
                    sub_sample, lumi_weight, regions, hists, config,
                )

                # Merge into accumulated results
                if merged is None:
                    merged = sub_results
                else:
                    for rn in sub_results:
                        for hn in sub_results[rn]:
                            merged[rn][hn] = merged[rn][hn] + sub_results[rn][hn]

            # Store merged results under the sample name
            for region_name in merged:
                if region_name not in results:
                    results[region_name] = {}
                results[region_name][sample_name] = merged[region_name]
                for hist_name, hdata in merged[region_name].items():
                    logger.info(
                        "  %s / %s / %s: merged integral = %.1f",
                        sample_name, region_name, hist_name, hdata.integral,
                    )

        else:
            # Single directory (or no lumi scaling) — process normally
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

            sub_results = _process_single_sample(
                sample, lumi_weight, regions, hists, config,
            )

            for region_name in sub_results:
                if region_name not in results:
                    results[region_name] = {}
                results[region_name][sample_name] = sub_results[region_name]

    return results
