#!/usr/bin/env python3
"""Generate a LaTeX longtable document from an ABCD results JSON file.

Usage:
    python abcd_results_table.py <results.json> [output.tex]

If output path is omitted, the .tex file is written alongside the JSON.
"""

import json
import sys
from pathlib import Path


REGION_LABELS = {
    "L1L1": "L1L1",
    "L1L2": "L1L2 (cross)",
    "L2L2": "L2L2",
    "L2L3": "L2L3 (cross)",
    "L3L3": "L3L3",
}


def make_tables(d):
    lines = []
    lines.append(r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
""")

    lumi = d["meta"]["luminosity_pb"]

    for reg, entries in d["regions"].items():
        label = REGION_LABELS.get(reg, reg)
        lines.append(
            f"\\begin{{longtable}}{{r r c c c c}}\n"
            f"\\caption{{ABCD background estimate --- {label} region. "
            f"Luminosity: {lumi:.2f}~pb$^{{-1}}$ (1\\% dataset).}}\n"
            f"\\label{{tab:abcd_{reg.lower()}}}\\\\\n"
            "\\hline\\hline\n"
            "$m$ [MeV] & SR width [MeV] & Prediction & Observed "
            "& 10\\% proj. & 100\\% proj. \\\\\n"
            "\\hline\\hline\n"
            "\\endfirsthead\n"
            "\\hline\\hline\n"
            "$m$ [MeV] & SR width [MeV] & Prediction & Observed "
            "& 10\\% proj. & 100\\% proj. \\\\\n"
            "\\hline\\hline\n"
            "\\endhead\n"
            "\\hline\n"
            "\\endfoot"
        )

        for e in entries:
            m     = int(round(e["mass_GeV"] * 1000))
            sw    = e["signal_window_GeV"] * 2 * 1000
            pred  = e["bkg_est"]
            pred_e = e["bkg_est_err"]
            obs   = e["n_a_obs"]
            obs_e = e["n_a_obs_err"]
            proj  = e["projections"]
            p10_label  = next(k for k in proj if "10" in k)
            p100_label = next(k for k in proj if "100" in k)
            p10   = proj[p10_label]["bkg_est"]
            p10e  = proj[p10_label]["bkg_est_err"]
            p100  = proj[p100_label]["bkg_est"]
            p100e = proj[p100_label]["bkg_est_err"]
            lines.append(
                f"  {m} & {sw:.2f} & ${pred:.1f} \\pm {pred_e:.1f}$ & "
                f"${obs:.0f} \\pm {obs_e:.1f}$ & "
                f"${p10:.1f} \\pm {p10e:.1f}$ & "
                f"${p100:.1f} \\pm {p100e:.1f}$ \\\\"
            )

        lines.append("\\end{longtable}\n")

    lines.append("\\end{document}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"ERROR: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else json_path.with_suffix(".tex")

    with open(json_path) as f:
        d = json.load(f)

    tex = make_tables(d)
    with open(out_path, "w") as f:
        f.write(tex)

    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
