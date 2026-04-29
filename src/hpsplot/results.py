"""Processing orchestration: samples x regions x histograms."""

import copy
import logging

import numpy as np

from .config import Config, SampleConfig
from .histogram import fill_histogram, fill_histogram_2d
from .region import Region
from .sample import Sample, compute_luminosity
from .signal_scaling import compute_signal_scale_factor, count_data_in_window
from .utils import safe_evaluate

logger = logging.getLogger(__name__)


_SCAN_COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _process_single_sample(sample, lumi_weight, regions, hists, config):
    """Load data once and fill histograms for every weight expression.

    When the sample has no ``weight_scan`` entries the single default weight
    is used and the return dict has one key (the sample name), matching the
    old behaviour.  When ``weight_scan`` is populated the ROOT data is loaded
    **once** and histograms are filled separately for each scan entry without
    re-reading from disk.

    Returns
    -------
    dict
        ``{virtual_name: {region_name: {hist_name: HistogramData}}}``

        *virtual_name* is the sample name for the no-scan case, or the
        ``WeightScanEntry.label`` for each scan entry.
    """
    hist_list = list(hists.values())
    region_list = list(regions.values())
    region_configs = [r.config for r in region_list]
    branches = sample.get_needed_branches(hist_list, region_configs)

    # Merge global aliases with per-sample overrides (sample wins on conflicts)
    effective_aliases = {**config.aliases, **sample.config.aliases}
    # load() automatically adds branches for weight_scan weight expressions
    data = sample.load(branches, aliases=effective_aliases)

    # Sample-level selection mask (applied identically for all weight entries)
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

    total_scale = sample.scale * lumi_weight
    # weight_scan_exprs falls back to [(sample.name, default_weight)] when no scan
    weight_exprs = sample.weight_scan_exprs

    all_results = {}
    for virtual_name, weight_expr in weight_exprs:
        sub_results = {}
        for region_name, region in regions.items():
            mask = region.apply(data)
            if sample_mask is not None:
                mask = mask & sample_mask

            sub_results[region_name] = {}
            for hist_name, hist_cfg in hists.items():
                values = safe_evaluate(hist_cfg.variable, data, mask=mask)
                weights = safe_evaluate(weight_expr, data, mask=mask)

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
                        virtual_name, region_name, hist_name, np.sum(hdata.contents),
                    )
                else:
                    hdata = fill_histogram(
                        values, weights,
                        hist_cfg.bins, hist_cfg.x_min, hist_cfg.x_max,
                    )
                    logger.info(
                        "  %s / %s / %s: integral = %.1f",
                        virtual_name, region_name, hist_name, hdata.integral,
                    )
                sub_results[region_name][hist_name] = hdata

        all_results[virtual_name] = sub_results

    return all_results


def _register_virtual_configs(sample, config):
    """Add virtual SampleConfig entries for weight_scan labels to config.samples.

    These lightweight entries carry color/label metadata so the plotting
    functions can render scan entries exactly like real samples.  The real
    sample (parent) is NOT removed — it remains in the list for non-scan use.
    Already-registered entries are skipped so calling this multiple times is safe.
    """
    if not sample.config.weight_scan:
        return
    existing_names = {s.name for s in config.samples}
    for i, entry in enumerate(sample.config.weight_scan):
        if entry.label in existing_names:
            continue
        color = entry.color or _SCAN_COLOR_CYCLE[i % len(_SCAN_COLOR_CYCLE)]
        virtual = SampleConfig(
            name=entry.label,
            label=entry.label,
            color=color,
            sample_type=sample.config.sample_type,
            weight=entry.weight,
            # No directory — this config is metadata-only; loading is via parent
        )
        config.samples.append(virtual)
        existing_names.add(entry.label)


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
    # Compute luminosity from data files + lumi file if configured.
    # Done before the histograms guard so ABCD-only configs also get the
    # correct luminosity on their canvas labels.
    if config.lumi_file:
        data_sample = next(
            (s for s in config.samples if s.sample_type == "data"), None
        )
        if data_sample:
            config.luminosity = compute_luminosity(
                data_sample.directories, config.lumi_file,
                run_min=data_sample.run_min, run_max=data_sample.run_max,
                exclude_runs=data_sample.exclude_runs,
            )
        else:
            logger.warning("lumi_file set but no data sample found — using luminosity=%.4g",
                           config.luminosity)

    # Nothing to do if there are no histograms to fill
    if not config.histograms:
        return {}

    # Build lookup maps
    sample_map = {s.name: s for s in config.samples}
    region_map = {r.name: r for r in config.regions}
    hist_map = {h.name: h for h in config.histograms}

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
                    scale = compute_signal_scale_factor(
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

    # Build a reverse map: scan entry label → parent SampleConfig.
    # This lets the plot reference scan labels directly in its samples list
    # while still loading the parent ROOT files only once.
    scan_label_to_parent = {}
    for s in config.samples:
        for entry in s.weight_scan:
            scan_label_to_parent[entry.label] = s

    # Build objects
    samples = {}
    for name in needed_samples:
        if name in sample_map:
            samples[name] = Sample(sample_map[name])
        elif name in scan_label_to_parent:
            parent_cfg = scan_label_to_parent[name]
            if parent_cfg.name not in samples:
                samples[parent_cfg.name] = Sample(parent_cfg)
        else:
            logger.debug("Sample '%s' referenced in plot but not in current sample set — skipping.", name)

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

                vn_results = _process_single_sample(
                    sub_sample, lumi_weight, regions, hists, config,
                )

                # Merge into accumulated results (keyed by virtual_name)
                if merged is None:
                    merged = vn_results
                else:
                    for vn in vn_results:
                        for rn in vn_results[vn]:
                            for hn in vn_results[vn][rn]:
                                merged[vn][rn][hn] = merged[vn][rn][hn] + vn_results[vn][rn][hn]

            # Store merged results
            for virtual_name, vn_sub in merged.items():
                for region_name, hist_results in vn_sub.items():
                    results.setdefault(region_name, {})[virtual_name] = hist_results
                    for hist_name, hdata in hist_results.items():
                        logger.info(
                            "  %s / %s / %s: merged integral = %.1f",
                            virtual_name, region_name, hist_name, hdata.integral,
                        )
            # Register any virtual SampleConfigs produced by weight_scan
            _register_virtual_configs(sample, config)

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

            vn_results = _process_single_sample(
                sample, lumi_weight, regions, hists, config,
            )

            for virtual_name, vn_sub in vn_results.items():
                for region_name, hist_results in vn_sub.items():
                    results.setdefault(region_name, {})[virtual_name] = hist_results
            # Register any virtual SampleConfigs produced by weight_scan
            _register_virtual_configs(sample, config)

    return results
