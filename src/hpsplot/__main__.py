"""CLI entry point: python -m hpsplot config.yaml [-o output_dir] [-v]"""

import argparse
import copy
import glob as globmod
import logging
import re
import sys

import json
from pathlib import Path

_RUN_RE = re.compile(r"_(\d{5,6})_")

from .config import load_config, RegionConfig
from .utils import extract_branch_names
from .plotting.style import set_hps_style
from .plotting.stack import plot_stack
from .plotting.overlay import plot_overlay
from .plotting.rad_frac import plot_rad_frac
from .plotting.smearing import plot_smearing, build_tool_json
from .plotting.abcd import (plot_abcd, plot_abcd_summary, plot_abcd_lumi_projections,
                             write_abcd_json, plot_abcd_aux_histogram)
from .plotting.binned import plot_binned_comparison
from .plotting.run_trend import plot_run_trend
from .results import process


def main():
    parser = argparse.ArgumentParser(
        prog="hpsplot",
        description="HPS analysis plotting tool",
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("-P", "--plots", nargs="+", default=None,
                        help="Only run these named plot(s) (comma- or space-separated)")
    parser.add_argument("-H", "--histograms", nargs="+", default=None,
                        help="Only plot these histogram(s) from the config")
    parser.add_argument("--logy", action="store_true",
                        help="Force log y-axis on all histograms")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--per-file", action="store_true",
                        help="Run the full pipeline independently for each data file, "
                             "writing output to per-run subdirectories")
    args = parser.parse_args()

    # Setup logging — only set our package to DEBUG, keep others at WARNING
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    pkg_logger = logging.getLogger("hpsplot")
    pkg_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    # Load config
    logger.info("Loading config: %s", args.config)
    config = load_config(args.config)

    if args.output_dir:
        config.output_dir = args.output_dir

    # Filter plots if --plots is specified
    if args.plots:
        plot_names = set()
        for item in args.plots:
            for name in item.split(","):
                name = name.strip()
                if name:
                    plot_names.add(name)
        available = {p.name for p in config.plots}
        unknown = plot_names - available
        if unknown:
            logger.error("Unknown plot(s): %s", ", ".join(sorted(unknown)))
            logger.info("Available: %s", ", ".join(sorted(available)))
            sys.exit(1)
        config.plots = [p for p in config.plots if p.name in plot_names]
        # Trim histograms to only those needed by the selected plots
        needed = {h for p in config.plots for h in p.histograms}
        config.histograms = [h for h in config.histograms if h.name in needed]

    # Filter histograms if --histograms is specified
    if args.histograms:
        # Support both space-separated (-H a b) and comma-separated (-H a,b)
        hist_names = set()
        for item in args.histograms:
            for name in item.split(","):
                name = name.strip()
                if name:
                    hist_names.add(name)
        available = {h.name for h in config.histograms}
        unknown = hist_names - available
        if unknown:
            logger.error("Unknown histogram(s): %s", ", ".join(sorted(unknown)))
            logger.info("Available: %s", ", ".join(sorted(available)))
            sys.exit(1)
        # Filter plot configs to only include requested histograms
        for plot in config.plots:
            plot.histograms = [h for h in plot.histograms if h in hist_names]

    # Override log_y if --logy is specified
    if args.logy:
        for h in config.histograms:
            h.log_y = True

    # Set style
    set_hps_style()

    if args.per_file:
        _run_per_file(config, args, logger)
        return

    _run_config(config, logger)


def _enumerate_data_runs(data_cfg):
    """Return (run_label, [filepaths]) pairs grouped by run number.

    Multiple files sharing the same run number (e.g. different segments) are
    collected into a single list so they are processed together.
    Respects run_min / run_max on the sample config if set.
    Files whose names contain no recognised run number are grouped under their stem.
    """
    run_min = data_cfg.run_min
    run_max = data_cfg.run_max
    exclude = set(data_cfg.exclude_runs or [])

    runs = {}   # run_label → [fpath, ...]
    for d in data_cfg.directories:
        if "*" in d:
            candidates = sorted(globmod.glob(d))
        else:
            p = Path(d)
            candidates = [str(p)] if p.suffix == ".root" else sorted(str(f) for f in p.glob("*.root"))
        for fpath in candidates:
            m = _RUN_RE.search(Path(fpath).name)
            if m:
                run = int(m.group(1))
                if run_min is not None and run < run_min:
                    continue
                if run_max is not None and run > run_max:
                    continue
                if run in exclude:
                    continue
                label = str(run)
            else:
                label = Path(fpath).stem
            runs.setdefault(label, []).append(fpath)

    # Return sorted by run label (numeric where possible)
    def _sort_key(lbl):
        try:
            return (0, int(lbl))
        except ValueError:
            return (1, lbl)

    return sorted(runs.items(), key=lambda kv: _sort_key(kv[0]))


def _scale_results(results, scale):
    """Return a copy of results with every HistogramData multiplied by scale."""
    scaled = {}
    for region_name, region_data in results.items():
        scaled[region_name] = {}
        for sample_name, sample_data in region_data.items():
            scaled[region_name][sample_name] = {
                hist_name: hdata * scale
                for hist_name, hdata in sample_data.items()
            }
    return scaled


def _load_hit_cat_fracs(path):
    """Parse a hit-category fraction file into {run_int: {cat: frac}}.

    Expected format (one run per line, comment lines start with #):
        run  L1L1  L1L2  L2L1  L2L2  Other
    """
    fracs = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            run = int(parts[0])
            fracs[run] = {
                "L1L1":  float(parts[1]),
                "L1L2":  float(parts[2]),
                "L2L1":  float(parts[3]),
                "L2L2":  float(parts[4]),
                "Other": float(parts[5]),
            }
    return fracs


def _hit_cat_sf(region_name, run_fracs, ref_fracs):
    """Return SF = frac(run)/frac(ref) for the hit category matching region_name.

    Category assignment (case-insensitive, from region name):
      * contains "L1L1" → L1L1 fraction
      * contains "L2L2" → L2L2 fraction
      * contains "L1L2" or "L2L1" → L1L2 + L2L1 combined fraction
      * otherwise → 1.0 (no per-category correction)
    """
    rn = region_name.upper()
    if "L1L1" in rn:
        num = run_fracs["L1L1"]
        den = ref_fracs["L1L1"]
    elif "L2L2" in rn:
        num = run_fracs["L2L2"]
        den = ref_fracs["L2L2"]
    elif "L1L2" in rn or "L2L1" in rn:
        num = run_fracs["L1L2"] + run_fracs["L2L1"]
        den = ref_fracs["L1L2"] + ref_fracs["L2L1"]
    else:
        return 1.0
    return num / den if den else 1.0


def _fit_hit_cat_polys(hit_cat_fracs, ref_fracs, order):
    """Fit degree-*order* polynomials of SF vs run number for each category.

    Returns a dict with keys "L1L1", "L1L2combined", "L2L2", "Other", each
    mapping to a numpy coefficient array suitable for np.polyval(coeffs, run).
    L1L2 and L2L1 are merged into a single "L1L2combined" fit using the
    combined (L1L2+L2L1) fraction relative to the reference.
    """
    import numpy as np
    runs = np.array(sorted(hit_cat_fracs.keys()))
    polys = {}
    for key, num_cats, den_val in [
        ("L1L1",        ["L1L1"],         ref_fracs["L1L1"]),
        ("L1L2combined",["L1L2", "L2L1"], ref_fracs["L1L2"] + ref_fracs["L2L1"]),
        ("L2L2",        ["L2L2"],         ref_fracs["L2L2"]),
        ("Other",       ["Other"],        ref_fracs["Other"]),
    ]:
        if den_val == 0:
            continue
        sf = np.array([
            sum(hit_cat_fracs[r][c] for c in num_cats) / den_val
            for r in runs
        ])
        polys[key] = np.polyfit(runs, sf, order)
    return polys


def _hit_cat_sf_poly(region_name, run_int, poly_coeffs):
    """Return the polynomial-evaluated SF for the category matching region_name."""
    import numpy as np
    rn = region_name.upper()
    if "L1L1" in rn:
        key = "L1L1"
    elif "L2L2" in rn:
        key = "L2L2"
    elif "L1L2" in rn or "L2L1" in rn:
        key = "L1L2combined"
    else:
        return 1.0
    coeffs = poly_coeffs.get(key)
    return float(np.polyval(coeffs, run_int)) if coeffs is not None else 1.0


def _run_per_file(config, args, logger):
    """Run the full pipeline once per run, grouping all files that share a run number."""
    data_cfg = next((s for s in config.samples if s.sample_type == "data"), None)
    if data_cfg is None:
        logger.error("--per-file requires a data sample.")
        sys.exit(1)

    runs = _enumerate_data_runs(data_cfg)
    logger.info("--per-file: %d run(s) found.", len(runs))

    base_output = args.output_dir or config.output_dir
    samples_map = {s.name: s for s in config.samples}

    # Load hit-category fractions if configured
    hit_cat_fracs = {}   # {run_int: {cat: frac}}
    ref_fracs = {}       # {cat: frac} for the reference run
    hit_cat_polys = {}   # {cat_key: poly_coeffs} — populated when poly_order >= 0
    if config.hit_cat_file and config.hit_cat_ref_run:
        hit_cat_fracs = _load_hit_cat_fracs(config.hit_cat_file)
        if config.hit_cat_ref_run not in hit_cat_fracs:
            logger.warning(
                "hit_cat_ref_run %d not found in %s — hit-category SF disabled.",
                config.hit_cat_ref_run, config.hit_cat_file,
            )
            hit_cat_fracs = {}
        else:
            ref_fracs = hit_cat_fracs[config.hit_cat_ref_run]
            if config.hit_cat_poly_order >= 0:
                hit_cat_polys = _fit_hit_cat_polys(
                    hit_cat_fracs, ref_fracs, config.hit_cat_poly_order
                )
                logger.info(
                    "Hit-category SFs: degree-%d polynomial fit to %d runs (ref=%d).",
                    config.hit_cat_poly_order, len(hit_cat_fracs), config.hit_cat_ref_run,
                )
            else:
                logger.info(
                    "Hit-category SFs: closest-run lookup; ref run %d: "
                    "L1L1=%.4f L1L2=%.4f L2L1=%.4f L2L2=%.4f",
                    config.hit_cat_ref_run,
                    ref_fracs["L1L1"], ref_fracs["L1L2"],
                    ref_fracs["L2L1"], ref_fracs["L2L2"],
                )

    # Identify run_trend plots that need cross-run accumulation
    run_trend_plots = [p for p in config.plots if p.plot_type == "run_trend"]
    # run_trend_records: plot_name → region_name → list of (run_label, {sample: yield})
    run_trend_records = {p.name: {r: [] for r in p.regions} for p in run_trend_plots}

    # Pre-compute MC histograms once at unit luminosity (lumi = 1.0 pb^-1).
    # Each per-run pass scales these by the actual run luminosity, avoiding
    # redundant ROOT file I/O for samples that don't change run-to-run.
    mc_cfg = copy.deepcopy(config)
    mc_cfg.samples = [s for s in mc_cfg.samples if s.sample_type != "data"]
    mc_cfg.luminosity = 1.0
    mc_cfg.lumi_file = ""  # lumi will be determined per-run from data
    logger.info("Pre-computing MC histograms at unit luminosity (cached for all runs)...")
    mc_results_unit = process(mc_cfg)

    for run_label, fpaths in runs:
        logger.info("--- per-run: %s (%d file(s)) ---", run_label, len(fpaths))

        # Build per-run config (data pointing at this run's files only)
        cfg = copy.deepcopy(config)
        for s in cfg.samples:
            if s.name == data_cfg.name:
                s.directories = fpaths
                s.run_min = None
                s.run_max = None
        cfg.output_dir = str(Path(base_output) / f"run_{run_label}")
        cfg.run_label = run_label

        # Process only data for this run.
        # process() also computes run_lumi via lumi_file → data_only_cfg.luminosity.
        data_only_cfg = copy.deepcopy(cfg)
        data_only_cfg.samples = [s for s in data_only_cfg.samples if s.sample_type == "data"]
        data_results = process(data_only_cfg)
        run_lumi = data_only_cfg.luminosity
        cfg.luminosity = run_lumi  # propagate for plot labels

        # Scale cached MC by this run's luminosity
        run_results = _scale_results(mc_results_unit, run_lumi)

        # Apply per-region hit-category SF to MC
        if hit_cat_fracs and ref_fracs:
            try:
                run_int = int(run_label)
            except ValueError:
                run_int = None
            if run_int is not None:
                for region_name in list(run_results.keys()):
                    if hit_cat_polys:
                        sf = _hit_cat_sf_poly(region_name, run_int, hit_cat_polys)
                    else:
                        closest = min(hit_cat_fracs.keys(), key=lambda r: abs(r - run_int))
                        sf = _hit_cat_sf(region_name, hit_cat_fracs[closest], ref_fracs)
                    if sf == 1.0:
                        continue
                    for sample_name, sample_data in run_results[region_name].items():
                        if samples_map.get(sample_name) and samples_map[sample_name].sample_type != "data":
                            run_results[region_name][sample_name] = {
                                h: hdata * sf for h, hdata in sample_data.items()
                            }

        # Merge data results
        for region_name, region_data in data_results.items():
            run_results.setdefault(region_name, {}).update(region_data)

        # Generate all per-run plots (stack, overlay, etc.) using merged results
        _generate_plots(cfg, run_results, logger)

        # Collect per-run yields for run_trend summary (all regions)
        for plot_cfg in run_trend_plots:
            if not plot_cfg.histograms or not plot_cfg.regions:
                continue
            hist_name = plot_cfg.histograms[0]
            for region_name in plot_cfg.regions:
                yields = {}
                for sample_name in plot_cfg.samples:
                    hdata = run_results.get(region_name, {}).get(sample_name, {}).get(hist_name)
                    yields[sample_name] = hdata.integral if hdata is not None else 0.0
                run_trend_records[plot_cfg.name][region_name].append((run_label, yields, run_lumi))

    # After all runs: produce one run_trend plot per region
    for plot_cfg in run_trend_plots:
        outdir = plot_cfg.output_dir or base_output
        for region_name in plot_cfg.regions:
            plot_run_trend(
                plot_cfg, run_trend_records[plot_cfg.name][region_name],
                region_name, samples_map, outdir, config.output_format,
            )


def _generate_plots(config, results, logger):
    """Generate all plots from a pre-filled results dict.

    Parameters
    ----------
    config : Config
    results : dict
        results[region_name][sample_name][hist_name] → HistogramData
    logger : logging.Logger
    """
    # Build lookup maps for plotting
    samples_map = {s.name: s for s in config.samples}
    region_map = {r.name: r for r in config.regions}
    hist_map = {h.name: h for h in config.histograms}

    # Load ANN model once if configured
    ann_scorer = None
    if config.ann is not None:
        from .ann_classifier import load_ann
        ann_model, ann_mean, ann_scale = load_ann(
            config.ann.model_path, config.ann.scaler_path
        )
        ann_scorer = (ann_model, ann_mean, ann_scale, config.ann.score_variable)
        logger.info("ANN scorer loaded: score_variable='%s'", config.ann.score_variable)

    # Generate plots
    logger.info("Generating plots...")
    for plot_cfg in config.plots:

        # run_trend plots are handled by _run_per_file after all runs complete
        if plot_cfg.plot_type == "run_trend":
            logger.debug("Skipping run_trend plot '%s' in per-run pass.", plot_cfg.name)
            continue

        # ABCD plots load their own data and don't use the histogram pipeline
        if plot_cfg.plot_type == "abcd":
            outdir = plot_cfg.output_dir or config.output_dir
            abcd_cfg = plot_cfg.abcd

            # --- aux variable: auto-generate per-bin regions ----------------
            # For each ABCDAuxConfig, each region in plot_cfg.regions is a
            # "base" region.  Per bin we synthesise a child RegionConfig that
            # appends the bin cut, named  {base}__auxvar__{var_clean}_{i}.
            # Children from all aux vars are processed in one shared pass so
            # the data_cache is reused across variables.
            #
            # aux_groups: aux_cfg → {base_name → [child_names]}
            original_base_regions = list(plot_cfg.regions)
            aux_groups = []   # list of (aux_cfg, {base_name: [child_region_name, ...]})
            all_child_regions = []  # flat list in processing order

            if abcd_cfg.aux_histograms:
                plot_cfg = copy.copy(plot_cfg)  # don't mutate shared config object
                for aux_cfg in abcd_cfg.aux_histograms:
                    if len(aux_cfg.bins) < 2:
                        logger.warning("aux_histogram variable '%s' has fewer than 2 bin edges — skipping.", aux_cfg.variable)
                        continue
                    var_clean = re.sub(r'[^a-zA-Z0-9]', '_', aux_cfg.variable).strip('_')
                    edges = aux_cfg.bins
                    groups_for_this_aux = {}
                    for base_name in original_base_regions:
                        base_cfg_obj = region_map.get(base_name)
                        if base_cfg_obj is None:
                            logger.warning("Base region '%s' not found, skipping.", base_name)
                            continue
                        children = []
                        for i in range(len(edges) - 1):
                            lo, hi = edges[i], edges[i + 1]
                            child_name = f"{base_name}__auxvar__{var_clean}_{i}"
                            child_sel  = (
                                f"({base_cfg_obj.selection}) & "
                                f"({aux_cfg.variable} >= {lo}) & "
                                f"({aux_cfg.variable} < {hi})"
                            )
                            child_lbl = f"${lo} \\leq {aux_cfg.variable} < {hi}$"
                            child_cfg = RegionConfig(name=child_name, selection=child_sel,
                                                     label=child_lbl)
                            config.regions.append(child_cfg)
                            region_map[child_name] = child_cfg
                            all_child_regions.append(child_name)
                            children.append(child_name)
                        groups_for_this_aux[base_name] = children
                    aux_groups.append((aux_cfg, groups_for_this_aux))
                # Replace the plot regions with the full set of child regions
                plot_cfg.regions = all_child_regions

            # Collect branches for all (possibly expanded) regions so each
            # sample is loaded once from disk and reused.
            all_region_branches = set()
            for rn in plot_cfg.regions:
                rc = region_map.get(rn)
                if rc:
                    all_region_branches |= extract_branch_names(rc.selection)
            for aux_cfg in abcd_cfg.aux_histograms:
                all_region_branches |= extract_branch_names(aux_cfg.variable)

            counts_cache = {}
            data_cache = {}  # sample_name → raw arrays; shared across all regions
            for region_name in plot_cfg.regions:
                region_cfg = region_map.get(region_name)
                if region_cfg is None:
                    logger.warning("Region '%s' not found, skipping.", region_name)
                    continue
                counts_cache[region_name] = plot_abcd(
                    plot_cfg, region_cfg, config, samples_map,
                    outdir, config.output_format,
                    ann_scorer=ann_scorer,
                    data_cache=data_cache,
                    all_region_branches=all_region_branches,
                )

            if aux_groups:
                # One histogram per (aux variable, base region) combination
                for aux_cfg, groups in aux_groups:
                    for base_name, children in groups.items():
                        base_cfg_obj = region_map.get(base_name)
                        bin_region_cfgs = [region_map[r] for r in children]
                        plot_abcd_aux_histogram(
                            plot_cfg, aux_cfg, base_cfg_obj, children,
                            bin_region_cfgs, config, samples_map,
                            outdir, config.output_format,
                            counts_cache=counts_cache,
                            data_cache=data_cache,
                        )
            else:
                # Standard summary bar chart (no aux variables defined)
                plot_abcd_summary(plot_cfg, config, samples_map, outdir, config.output_format,
                                  counts_cache=counts_cache, ann_scorer=ann_scorer)

            if abcd_cfg.lumi_projections:
                plot_abcd_lumi_projections(plot_cfg, config, samples_map, outdir,
                                           config.output_format, counts_cache=counts_cache,
                                           ann_scorer=ann_scorer)
            if abcd_cfg.json_output:
                write_abcd_json(plot_cfg, config, samples_map, counts_cache,
                                abcd_cfg.json_output)
            continue

        # Binned comparison plots iterate over regions but not histograms
        if plot_cfg.plot_type == "binned":
            outdir = plot_cfg.output_dir or config.output_dir
            if not plot_cfg.regions:
                plot_binned_comparison(plot_cfg, None, config, samples_map,
                                       outdir, config.output_format)
            else:
                for region_name in plot_cfg.regions:
                    region_cfg = region_map.get(region_name)
                    if region_cfg is None:
                        logger.warning("Region '%s' not found, skipping.", region_name)
                        continue
                    plot_binned_comparison(plot_cfg, region_cfg, config, samples_map,
                                           outdir, config.output_format)
            continue

        smearing_results = {}  # accumulated per smearing plot for JSON output

        for region_name in plot_cfg.regions:
            region_cfg = region_map.get(region_name)
            if region_cfg is None:
                logger.warning("Region '%s' not found, skipping.", region_name)
                continue

            for hist_name in plot_cfg.histograms:
                hist_cfg = hist_map.get(hist_name)
                if hist_cfg is None:
                    logger.warning("Histogram '%s' not found, skipping.", hist_name)
                    continue

                outdir = plot_cfg.output_dir or config.output_dir
                if plot_cfg.plot_type == "stack":
                    plot_stack(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, outdir, config.output_format,
                        run_label=config.run_label,
                    )
                elif plot_cfg.plot_type == "overlay":
                    plot_overlay(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, outdir, config.output_format,
                        run_label=config.run_label,
                    )
                elif plot_cfg.plot_type == "rad_frac":
                    plot_rad_frac(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, outdir, config.output_format,
                        run_label=config.run_label,
                    )
                elif plot_cfg.plot_type == "smearing":
                    result = plot_smearing(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, outdir, config.output_format,
                        beam_energy=config.beam_energy,
                        run_label=config.run_label,
                    )
                    if result is not None:
                        smearing_results[f"{region_name}/{hist_name}"] = result

        # Write diagnostic JSON output after all histograms/regions for this smearing plot
        if (plot_cfg.plot_type == "smearing"
                and plot_cfg.smearing is not None
                and plot_cfg.smearing.json_output
                and smearing_results):
            json_path = Path(plot_cfg.smearing.json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(smearing_results, f, indent=4)
            logger.info("Smearing JSON written: %s", json_path)

        # Write TrackSmearingTool-compatible JSON (merges with existing file)
        if (plot_cfg.plot_type == "smearing"
                and plot_cfg.smearing is not None
                and plot_cfg.smearing.tool_json_output
                and smearing_results):
            tool_data = build_tool_json(plot_cfg.smearing, smearing_results, hist_map)
            if tool_data:
                tool_path = Path(plot_cfg.smearing.tool_json_output)
                existing = {}
                if tool_path.exists():
                    with open(tool_path) as f:
                        existing = json.load(f)
                existing.update(tool_data)
                tool_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tool_path, "w") as f:
                    json.dump(existing, f, indent=4)
                logger.info("TrackSmearingTool JSON written: %s", tool_path)

    logger.info("Done.")


def _run_config(config, logger):
    """Process one config end-to-end: fill histograms and generate all plots.

    Returns
    -------
    dict
        results[region_name][sample_name][hist_name] → HistogramData
    """
    logger.info("Processing samples...")
    results = process(config)
    _generate_plots(config, results, logger)
    return results


if __name__ == "__main__":
    main()
