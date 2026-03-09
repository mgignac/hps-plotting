"""CLI entry point: python -m hpsplot config.yaml [-o output_dir] [-v]"""

import argparse
import logging
import sys

import json
from pathlib import Path

from .config import load_config
from .plotting.style import set_hps_style
from .plotting.stack import plot_stack
from .plotting.overlay import plot_overlay
from .plotting.rad_frac import plot_rad_frac
from .plotting.smearing import plot_smearing, build_tool_json
from .plotting.abcd import plot_abcd, plot_abcd_summary, plot_abcd_summary
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

    # Process: fill all histograms
    logger.info("Processing samples...")
    results = process(config)

    # Build lookup maps for plotting
    samples_map = {s.name: s for s in config.samples}
    region_map = {r.name: r for r in config.regions}
    hist_map = {h.name: h for h in config.histograms}

    # Generate plots
    logger.info("Generating plots...")
    for plot_cfg in config.plots:

        # ABCD plots load their own data and don't use the histogram pipeline
        if plot_cfg.plot_type == "abcd":
            outdir = plot_cfg.output_dir or config.output_dir
            counts_cache = {}
            for region_name in plot_cfg.regions:
                region_cfg = region_map.get(region_name)
                if region_cfg is None:
                    logger.warning("Region '%s' not found, skipping.", region_name)
                    continue
                counts_cache[region_name] = plot_abcd(
                    plot_cfg, region_cfg, config, samples_map,
                    outdir, config.output_format,
                )
            plot_abcd_summary(plot_cfg, config, samples_map, outdir, config.output_format,
                              counts_cache=counts_cache)
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
                    )
                elif plot_cfg.plot_type == "overlay":
                    plot_overlay(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, outdir, config.output_format,
                    )
                elif plot_cfg.plot_type == "rad_frac":
                    plot_rad_frac(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, outdir, config.output_format,
                    )
                elif plot_cfg.plot_type == "smearing":
                    result = plot_smearing(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, outdir, config.output_format,
                        beam_energy=config.beam_energy,
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


if __name__ == "__main__":
    main()
