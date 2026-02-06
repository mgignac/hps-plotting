"""CLI entry point: python -m hpsplot config.yaml [-o output_dir] [-v]"""

import argparse
import logging
import sys

from .config import load_config
from .plotting.style import set_hps_style
from .plotting.stack import plot_stack
from .plotting.overlay import plot_overlay
from .plotting.rad_frac import plot_rad_frac
from .results import process


def main():
    parser = argparse.ArgumentParser(
        prog="hpsplot",
        description="HPS analysis plotting tool",
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Override output directory")
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

    # Filter histograms if --histograms is specified
    if args.histograms:
        hist_names = set(args.histograms)
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

                if plot_cfg.plot_type == "stack":
                    plot_stack(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, config.output_dir, config.output_format,
                    )
                elif plot_cfg.plot_type == "overlay":
                    plot_overlay(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, config.output_dir, config.output_format,
                    )
                elif plot_cfg.plot_type == "rad_frac":
                    plot_rad_frac(
                        plot_cfg, hist_cfg, region_cfg, results,
                        samples_map, config.output_dir, config.output_format,
                    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
