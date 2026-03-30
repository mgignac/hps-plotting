#!/usr/bin/env python3
"""Generate a LaTeX table of run numbers and luminosities for the 1% data pass."""

import glob
import re
import math
from pathlib import Path

DATA_DIR  = "/sdf/data/hps/physics2021/preselection/v7/data_1pc"
LUMI_FILE = "/sdf/data/hps/physics2021/preselection/v2/data/lumi_batch1.txt"
N_COLS    = 4   # number of (Run, L) column pairs

# --- collect run numbers present in data directory --------------------------
run_re = re.compile(r"_(\d{5,6})_")
runs_in_dir = set()
for f in glob.glob(f"{DATA_DIR}/*.root"):
    m = run_re.search(Path(f).name)
    if m:
        runs_in_dir.add(int(m.group(1)))

# --- parse luminosity file --------------------------------------------------
lumi = {}
with open(LUMI_FILE) as fh:
    for line in fh:
        m = re.match(r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+(\S+)", line)
        if m and m.group(2) != "N/A":
            lumi[int(m.group(1))] = float(m.group(2))

# --- build sorted list of (run, luminosity) pairs ---------------------------
rows = [(r, lumi[r]) for r in sorted(runs_in_dir) if r in lumi]
total = sum(l for _, l in rows)
n_rows = len(rows)

# pad to a multiple of N_COLS so columns are equal length
n_per_col = math.ceil(n_rows / N_COLS)
rows += [(None, None)] * (n_per_col * N_COLS - n_rows)

# --- build LaTeX table ------------------------------------------------------
col_spec = " | ".join(["r r"] * N_COLS)
header   = " & ".join([r"Run & $\mathcal{L}$~[pb$^{-1}$]"] * N_COLS)

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(
    r"\caption{Run numbers and integrated luminosities for the 2021 HPS dataset "
    r"used in this analysis (1\% pre-scale pass). "
    r"Runs with no valid entry in the run-summary file are excluded. "
    rf"The total integrated luminosity across all {n_rows} runs is "
    rf"$\mathcal{{L}} = {total:.4f}$~pb$^{{-1}}$.}}"
)
lines.append(r"\label{tab:runs}")
lines.append(r"\setlength{\tabcolsep}{7pt}")
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
lines.append(rf"\multicolumn{{{2 * N_COLS}}}{{r}}{{Total: ${total:.4f}$~pb$^{{-1}}$}} \\")
lines.append(r"\hline\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

print("\n".join(lines))
