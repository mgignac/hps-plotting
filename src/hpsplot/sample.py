"""ROOT file reading via uproot."""

import json
import logging
import re
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .config import SampleConfig
from .utils import extract_branch_names

logger = logging.getLogger(__name__)

# Pattern to extract run number from data filenames like merged_hps_014187_job1.root
_RUN_RE = re.compile(r"_(\d{5,6})_")


def parse_lumi_file(lumi_path):
    """Parse a run-by-run luminosity text file.

    Expected format (whitespace-delimited, header line, dashes separator):
        Run  Found  Expected  Fraction  Luminosity
        -------
        14185  42  827  5.08%  0.018638
        14187   0  N/A    N/A       N/A

    Returns
    -------
    dict
        Mapping of run number (int) to luminosity (float).
        Runs with N/A luminosity are omitted.
    """
    run_lumi = {}
    with open(lumi_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("Run", "-", "Total", "Runs")):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                run = int(parts[0])
            except ValueError:
                continue
            if parts[4] == "N/A":
                continue
            try:
                lumi = float(parts[4])
            except ValueError:
                continue
            run_lumi[run] = lumi
    return run_lumi


def compute_luminosity(data_dir, lumi_file):
    """Compute total luminosity by matching data files to a run-lumi table.

    Scans ROOT files in *data_dir*, extracts run numbers from their filenames,
    then sums the corresponding luminosities from *lumi_file*.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing data ROOT files.
    lumi_file : str or Path
        Path to the run-by-run luminosity text file.

    Returns
    -------
    float
        Total integrated luminosity for the runs present in *data_dir*.
    """
    data_dir = Path(data_dir)
    lumi_file = Path(lumi_file)

    if not lumi_file.exists():
        logger.warning("Lumi file not found: %s", lumi_file)
        return 0.0

    run_lumi = parse_lumi_file(lumi_file)

    # Extract unique run numbers from data filenames
    runs_found = set()
    if data_dir.suffix == ".root":
        # Single file — extract run number from its name
        m = _RUN_RE.search(data_dir.name)
        if m:
            runs_found.add(int(m.group(1)))
    else:
        for root_file in sorted(data_dir.glob("*.root")):
            m = _RUN_RE.search(root_file.name)
            if m:
                runs_found.add(int(m.group(1)))

    # Sum luminosities
    total_lumi = 0.0
    missing = []
    for run in sorted(runs_found):
        if run in run_lumi:
            total_lumi += run_lumi[run]
        else:
            missing.append(run)

    if missing:
        logger.warning(
            "Runs in data but not in lumi file (or N/A): %s",
            ", ".join(str(r) for r in missing),
        )

    logger.info(
        "Luminosity from %d runs: %.6f (scanned %s)",
        len(runs_found) - len(missing), total_lumi, data_dir,
    )
    return total_lumi


class Sample:
    """Wraps a SampleConfig and provides data loading from ROOT files."""

    def __init__(self, config: SampleConfig):
        self.config = config
        self.name = config.name
        self.sample_type = config.sample_type
        self.label = config.label
        self.color = config.color
        self.weight_expr = config.weight
        self.scale = config.scale
        self._data = None

    def load(self, branches, aliases=None):
        """Load specified branches from all ROOT files in the sample directory.

        Parameters
        ----------
        branches : set or list
            Branch names to read from the TTree (may include alias names).
        aliases : dict, optional
            Mapping of alias name to full ROOT branch path.

        Returns
        -------
        dict
            Mapping of branch name to numpy array.
        """
        path = Path(self.config.directory)
        tree = self.config.tree
        aliases = aliases or {}

        if path.suffix == ".root":
            pattern = f"{path}:{tree}"
        else:
            pattern = f"{path}/*.root:{tree}"

        branches = sorted(set(branches))

        # Separate aliased branches from direct branches
        aliased = [b for b in branches if b in aliases]
        direct = [b for b in branches if b not in aliases]

        logger.info("Loading sample '%s' from %s", self.name, pattern)
        logger.debug("  branches: %s", branches)
        if aliased:
            logger.debug("  aliases: %s", {a: aliases[a] for a in aliased})

        # Resolve aliases: separate TVector3 component aliases (ending in .fX/.fY/.fZ)
        # from regular aliases that uproot can handle via its expressions engine
        tvec_aliases = {}   # alias_name -> (full_branch_path, component)
        expr_aliases = {}   # alias_name -> full_branch_path (for uproot expressions)
        for a in aliased:
            full_path = aliases[a]
            if full_path.endswith((".fX", ".fY", ".fZ")):
                component = full_path.rsplit(".", 1)[1]  # "fX", "fY", or "fZ"
                branch_path = full_path.rsplit(".", 1)[0]  # path without .fX
                tvec_aliases[a] = (branch_path, component)
            else:
                expr_aliases[a] = full_path

        # Everything except TVector3 components goes into expressions
        expr_branches = list(expr_aliases.keys()) + direct

        # TVector3 parent branches (paths with /) — read separately via filter_name
        tvec_parents = set()
        for branch_path, _ in tvec_aliases.values():
            tvec_parents.add(branch_path)

        try:
            # Main read: all scalar branches via expressions
            if expr_branches:
                arrays = uproot.concatenate(
                    pattern,
                    expressions=expr_branches,
                    aliases=expr_aliases if expr_aliases else None,
                    library="ak",
                )
            else:
                arrays = None

            # Separate read for TVector3 parent branches
            tvec_arrays = None
            if tvec_parents:
                tvec_parents_set = set(tvec_parents)
                tvec_arrays = uproot.concatenate(
                    pattern,
                    filter_name=lambda name: name in tvec_parents_set,
                    library="ak",
                )
        except Exception as e:
            logger.error("Failed to load sample '%s': %s", self.name, e)
            raise

        # Convert awkward arrays to numpy
        self._data = {}
        for b in branches:
            if b in tvec_aliases:
                branch_path, component = tvec_aliases[b]
                short_name = branch_path.rsplit("/", 1)[-1] if "/" in branch_path else branch_path
                vec = tvec_arrays[short_name]
                self._data[b] = ak.to_numpy(getattr(vec, component))
            else:
                self._data[b] = ak.to_numpy(arrays[b])

        logger.info("  Loaded %d events", len(next(iter(self._data.values()))))
        return self._data

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError(f"Sample '{self.name}' has not been loaded yet.")
        return self._data

    def get_lumi_weight(self, luminosity):
        """Compute luminosity scale factor from summary.json.

        Reads summary.json from the sample directory and computes:
            cross_section * luminosity / n_events

        where n_events = total_input_files * n_events_per_file.

        Parameters
        ----------
        luminosity : float
            Integrated luminosity.

        Returns
        -------
        tuple (lumi_weight, cross_section, n_events)
            lumi_weight : float — scale factor to multiply per-event weights by.
            cross_section : float — from summary.json (0 on fallback).
            n_events : int — generated events (0 on fallback).
        """
        fallback = (1.0, 0.0, 0)

        path = Path(self.config.directory)
        # For single-file samples, look for summary.json in the parent dir
        if path.suffix == ".root":
            summary_path = path.parent / "summary.json"
        else:
            summary_path = path / "summary.json"

        if not summary_path.exists():
            logger.warning(
                "summary.json not found for sample '%s' at %s — skipping lumi scaling",
                self.name, summary_path,
            )
            return fallback

        with open(summary_path) as f:
            summary = json.load(f)

        # Support both flat and nested (under "summary" key) layouts
        if "summary" in summary and isinstance(summary["summary"], dict):
            summary = summary["summary"]

        cross_section = summary.get("cross_section")
        total_input_files = summary.get("total_input_files")
        n_events_per_file = summary.get("n_events_per_file")

        if cross_section is None or total_input_files is None or n_events_per_file is None:
            missing = [k for k in ("cross_section", "total_input_files", "n_events_per_file")
                       if summary.get(k) is None]
            logger.warning(
                "summary.json for sample '%s' missing keys: %s — skipping lumi scaling",
                self.name, ", ".join(missing),
            )
            return fallback

        n_events = total_input_files * n_events_per_file
        if n_events <= 0:
            logger.warning(
                "n_events = 0 for sample '%s' — skipping lumi scaling", self.name,
            )
            return fallback

        lumi_weight = cross_section * luminosity / n_events
        return lumi_weight, cross_section, n_events

    def get_needed_branches(self, histogram_configs, region_configs):
        """Determine all branches needed for this sample's histograms, regions, and weights.

        Parameters
        ----------
        histogram_configs : list of HistogramConfig
        region_configs : list of RegionConfig

        Returns
        -------
        set
            Branch names to read.
        """
        branches = set()

        for h in histogram_configs:
            branches |= extract_branch_names(h.variable)

        for r in region_configs:
            branches |= extract_branch_names(r.selection)

        branches |= extract_branch_names(self.weight_expr)

        return branches
