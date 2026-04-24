"""Plot hit-category scale factors (frac(run)/frac(ref)) vs run number.

Usage:
    python paper_scripts/plot_hit_cat_sf.py [--order N] [--output path.pdf]

Only runs that appear in both the 5pc file listing AND the hit-cat table are shown.
A polynomial of degree --order is overlaid on each category panel.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_5PC_DIR = Path("/sdf/data/hps/physics2021/preselection/v8/data_5pc")
HIT_CAT_FILE = Path("/sdf/data/hps/users/mgignac/software/2021-ana/plotting/data/hit_cat_frac.txt")
REF_RUN      = 14268
OUTPUT_DEFAULT = "plots/hit_cat_sf.pdf"

CATEGORIES = ["L1L1", "L1L2", "L2L1", "L2L2", "Other"]
COLORS      = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

_RUN_RE = re.compile(r"_(\d{5,6})_")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hit_cat_fracs(path):
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


def extract_runs_from_dir(directory):
    """Return sorted list of unique integer run numbers found in directory."""
    runs = set()
    for p in Path(directory).glob("*.root"):
        m = _RUN_RE.search(p.name)
        if m:
            runs.add(int(m.group(1)))
    return sorted(runs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--order", type=int, default=None,
                        help="Polynomial degree to fit (default: show data only, prompt for order)")
    parser.add_argument("--output", default=OUTPUT_DEFAULT,
                        help=f"Output PDF path (default: {OUTPUT_DEFAULT})")
    parser.add_argument("--ref-run", type=int, default=REF_RUN,
                        help=f"Reference run number (default: {REF_RUN})")
    args = parser.parse_args()

    # Load data
    fracs = load_hit_cat_fracs(HIT_CAT_FILE)
    runs_5pc = set(extract_runs_from_dir(DATA_5PC_DIR))

    if args.ref_run not in fracs:
        sys.exit(f"Reference run {args.ref_run} not found in {HIT_CAT_FILE}")
    ref = fracs[args.ref_run]

    # Keep only runs present in both the hit-cat table and the 5pc directory
    common_runs = sorted(r for r in fracs if r in runs_5pc)
    print(f"Runs in hit-cat table : {len(fracs)}")
    print(f"Runs in v8/data_5pc   : {len(runs_5pc)}")
    print(f"Common runs           : {len(common_runs)}")

    if args.ref_run not in runs_5pc:
        print(f"WARNING: reference run {args.ref_run} is NOT in data_5pc directory")

    x = np.array(common_runs)
    # SF per category
    sfs = {}
    for cat in CATEGORIES:
        sfs[cat] = np.array([
            fracs[r][cat] / ref[cat] if ref[cat] > 0 else np.nan
            for r in common_runs
        ])

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    n_cats = len(CATEGORIES)
    fig = plt.figure(figsize=(14, 3.2 * n_cats))
    gs = gridspec.GridSpec(n_cats, 1, hspace=0.45)

    poly_coeffs = {}
    for i, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
        ax = fig.add_subplot(gs[i])
        y = sfs[cat]
        mask = np.isfinite(y)

        ax.scatter(x[mask], y[mask], s=20, color=color, zorder=3, label=cat)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(args.ref_run, color="gray", linestyle=":", linewidth=0.8,
                   label=f"ref run {args.ref_run}")

        if args.order is not None and mask.sum() >= args.order + 1:
            coeffs = np.polyfit(x[mask], y[mask], args.order)
            poly_coeffs[cat] = coeffs
            x_fit = np.linspace(x[mask].min(), x[mask].max(), 500)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, color=color, linewidth=1.5,
                    label=f"deg-{args.order} poly")
            # Print coeffs
            terms = " + ".join(
                f"{c:.4g}·x^{args.order - j}" for j, c in enumerate(coeffs)
            )
            print(f"{cat}: {terms}")

        ax.set_ylabel(f"SF ({cat})")
        ax.set_xlabel("Run number")
        ax.set_title(f"{cat}:  frac(run) / frac(ref={args.ref_run})")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(x.min() - 20, x.max() + 20)

        # Tick every run
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in x], rotation=90, fontsize=5)

    fig.suptitle("Hit-category scale factors vs run number", fontsize=12, y=1.01)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
