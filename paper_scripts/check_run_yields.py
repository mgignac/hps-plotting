#!/usr/bin/env python3
"""Scan a run_trend JSON file and report runs matching filter criteria.

By default, flags runs whose data/MC ratio is outside [0.5, 1.5].

Usage
-----
    python check_run_yields.py <json_file> [options]

Examples
--------
    # Default: flag runs with data/MC outside [0.5, 1.5]
    python check_run_yields.py plots/run_trend/yield_vs_run_loose.json

    # Custom ratio window
    python check_run_yields.py plots/run_trend/yield_vs_run_loose.json --ratio-min 0.8 --ratio-max 1.2

    # Runs with data yield below a threshold
    python check_run_yields.py plots/run_trend/yield_vs_run_loose.json --min-data 100

    # Runs with very low lumi
    python check_run_yields.py plots/run_trend/yield_vs_run_loose.json --min-lumi 0.001

    # Print all runs regardless
    python check_run_yields.py plots/run_trend/yield_vs_run_loose.json --all
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Identify problematic runs from a run_trend JSON file.")
    parser.add_argument("json_file", help="Path to the run_trend JSON file.")
    parser.add_argument("--min-data", type=float, default=None,
                        help="Flag runs with data yield strictly below this value (default: flag zeros only).")
    parser.add_argument("--ratio-min", type=float, default=0.5,
                        help="Flag runs with data/MC ratio below this value (default: 0.5).")
    parser.add_argument("--ratio-max", type=float, default=1.5,
                        help="Flag runs with data/MC ratio above this value (default: 1.5).")
    parser.add_argument("--min-lumi", type=float, default=None,
                        help="Flag runs with luminosity below this value.")
    parser.add_argument("--all", action="store_true",
                        help="Print all runs, not just flagged ones.")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    # Sort by run number (numeric where possible)
    def sort_key(k):
        try:
            return int(k)
        except ValueError:
            return k

    runs = sorted(data.keys(), key=sort_key)

    check_zero_data = False  # ratio window is now the default filter

    flagged = []
    for run in runs:
        entry = data[run]
        lumi       = entry.get("lumi", 0.0)
        data_yield = entry.get("data", 0.0)
        mc_total   = entry.get("mc_total", 0.0)
        ratio      = entry.get("data_mc_ratio", None)
        bgs        = entry.get("backgrounds", {})

        reasons = []

        if check_zero_data and data_yield == 0:
            reasons.append("data=0")

        if args.min_data is not None and data_yield < args.min_data:
            reasons.append(f"data={data_yield:.1f} < {args.min_data}")

        if args.min_lumi is not None and lumi < args.min_lumi:
            reasons.append(f"lumi={lumi:.4g} < {args.min_lumi}")

        if args.ratio_min is not None:
            if ratio is None:
                reasons.append("no ratio (MC=0)")
            elif ratio < args.ratio_min:
                reasons.append(f"ratio={ratio:.3f} < {args.ratio_min}")

        if args.ratio_max is not None:
            if ratio is not None and ratio > args.ratio_max:
                reasons.append(f"ratio={ratio:.3f} > {args.ratio_max}")

        if reasons or args.all:
            bg_str = "  ".join(f"{k}={v:.1f}" for k, v in bgs.items())
            ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
            flag_str = f"  *** {', '.join(reasons)}" if reasons else ""
            print(f"Run {run:>8s}  lumi={lumi:.4g}  data={data_yield:.1f}  "
                  f"mc={mc_total:.1f}  ratio={ratio_str}  [{bg_str}]{flag_str}")
            if reasons:
                flagged.append(run)

    print()
    if flagged:
        print(f"{len(flagged)} / {len(runs)} runs flagged:")
        print(" ".join(flagged))
    else:
        print(f"No runs flagged out of {len(runs)} total.")


if __name__ == "__main__":
    main()
