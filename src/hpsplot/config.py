"""YAML configuration loading into dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class SampleConfig:
    name: str
    directory: str
    sample_type: str  # "data", "background", "signal"
    label: str = ""
    color: str = "black"
    weight: str = "1.0"
    scale: float = 1.0
    lumi_scale: bool = False  # if True, compute weight from summary.json
    tree: str = ""  # per-sample override
    rad_frac_role: str = ""  # "numerator" or "denominator" for radiative fraction

    def __post_init__(self):
        if not self.label:
            self.label = self.name
        if self.sample_type not in ("data", "background", "signal"):
            raise ValueError(
                f"Invalid sample_type '{self.sample_type}' for sample '{self.name}'. "
                "Must be 'data', 'background', or 'signal'."
            )
        if self.rad_frac_role and self.rad_frac_role not in ("numerator", "denominator"):
            raise ValueError(
                f"Invalid rad_frac_role '{self.rad_frac_role}' for sample '{self.name}'. "
                "Must be 'numerator', 'denominator', or empty."
            )


@dataclass
class RegionConfig:
    name: str
    selection: str
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.name


@dataclass
class HistogramConfig:
    name: str
    variable: str
    bins: int = 50
    x_min: float = 0.0
    x_max: float = 1.0
    x_label: str = ""
    y_label: str = "Events"
    log_y: bool = False


@dataclass
class PlotConfig:
    name: str
    plot_type: str  # "stack" or "overlay"
    histograms: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    samples: List[str] = field(default_factory=list)
    data_sample: str = ""
    signal_samples: List[str] = field(default_factory=list)
    normalize: bool = False
    normalize_to_data: bool = False
    background_fractions: Dict[str, float] = field(default_factory=dict)
    signal_fractions: Dict[str, float] = field(default_factory=dict)
    ratio_y_min: float = 0.5
    ratio_y_max: float = 1.5

    def __post_init__(self):
        if self.plot_type not in ("stack", "overlay", "rad_frac"):
            raise ValueError(
                f"Invalid plot_type '{self.plot_type}' for plot '{self.name}'. "
                "Must be 'stack', 'overlay', or 'rad_frac'."
            )


@dataclass
class Config:
    samples: List[SampleConfig] = field(default_factory=list)
    regions: List[RegionConfig] = field(default_factory=list)
    histograms: List[HistogramConfig] = field(default_factory=list)
    plots: List[PlotConfig] = field(default_factory=list)
    output_dir: str = "plots/"
    output_format: str = "pdf"
    tree: str = ""
    luminosity: float = 1.0
    lumi_file: str = ""
    aliases: Dict[str, str] = field(default_factory=dict)


def load_config(path: str) -> Config:
    """Load a YAML configuration file and return a Config object."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = Config(
        output_dir=raw.get("output_dir", "plots/"),
        output_format=raw.get("output_format", "pdf"),
        tree=raw.get("tree", ""),
        luminosity=raw.get("luminosity", 1.0),
        lumi_file=raw.get("lumi_file", ""),
        aliases=raw.get("aliases", {}),
    )

    for s in raw.get("samples", []):
        sample = SampleConfig(**s)
        if not sample.tree:
            sample.tree = config.tree
        config.samples.append(sample)

    for r in raw.get("regions", []):
        config.regions.append(RegionConfig(**r))

    for h in raw.get("histograms", []):
        config.histograms.append(HistogramConfig(**h))

    for p in raw.get("plots", []):
        config.plots.append(PlotConfig(**p))

    # Auto-populate samples for rad_frac plots from rad_frac_role annotations
    rad_frac_samples = [s.name for s in config.samples if s.rad_frac_role]
    for plot in config.plots:
        if plot.plot_type == "rad_frac" and not plot.samples:
            plot.samples = list(rad_frac_samples)

    return config
