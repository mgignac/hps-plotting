#!/usr/bin/env python3
"""Generate a LaTeX table of run numbers and combined luminosities from pass-5 v9 batches."""

import re
import math

LUMI_FILES = [
    "/sdf/data/hps/physics2021/data/recon/pass5_v9/merged/merge-batch-1/lumi.txt",
    "/sdf/data/hps/physics2021/data/recon/pass5_v9/merged/merge-batch-2/lumi.txt",
]
N_COLS = 6


def parse_lumi(path):
    lumi = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"^\s*(\d+)\s+\S+\s+\S+\s+\S+\s+(\S+)", line)
            if m and m.group(2) != "N/A":
                lumi[int(m.group(1))] = float(m.group(2))
    return lumi


# Sum luminosities across batches
batches = [parse_lumi(p) for p in LUMI_FILES]
all_runs = sorted(set().union(*batches))
combined = {r: sum(b.get(r, 0.0) for b in batches) for r in all_runs}
total = sum(combined.values())
n_rows = len(all_runs)

# Pad to equal-length columns
rows = [(r, combined[r]) for r in all_runs]
n_per_col = math.ceil(n_rows / N_COLS)
rows += [(None, None)] * (n_per_col * N_COLS - n_rows)

# Build LaTeX
col_spec = " | ".join(["r r"] * N_COLS)
hdr_cell = r"Run & $\mathcal{L}$~[pb$^{-1}$]"
header   = " & ".join([hdr_cell] * N_COLS)

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(
    r"\caption{Run numbers and integrated luminosities from the 2021 HPS full "
    r"reconstruction (pass-5 v9, merge batches 1 and 2 combined). "
    r"Luminosities from the two batches are summed per run. "
    rf"The total integrated luminosity across all {n_rows} runs is "
    rf"$\mathcal{{L}} = {total:.4f}$~pb$^{{-1}}$.}}"
)
lines.append(r"\label{tab:runs_full}")
lines.append(r"\setlength{\tabcolsep}{5pt}")
lines.append(r"\small")
lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\hline\hline")
lines.append(header + r" \\")
lines.append(r"\hline")

for i in range(n_per_col):
    cells = []
    for j in range(N_COLS):
        run, lum = rows[i + j * n_per_col]
        if run is None:
            cells.extend(["", ""])
        else:
            cells.append(str(run))
            cells.append(f"{lum:.5f}")
    lines.append(" & ".join(cells) + r" \\")

lines.append(r"\hline")
lines.append(
    rf"\multicolumn{{{2 * N_COLS}}}{{r}}"
    rf"{{Total: ${total:.4f}$~pb$^{{-1}}$}} \\"
)
lines.append(r"\hline\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

print("\n".join(lines))
