"""YAML configuration loading into dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


@dataclass
class Eps2ScanConfig:
    """Auto-generate lifetime-reweighted signal yields for a grid of ε² values.

    cτ [mm] = 8.109×10⁻⁸ / (ε² × m_A' [MeV])
    Weight per event: base_weight × gen_length_mm × exp(−Δz/(βγ·cτ)) / (βγ·cτ)
    where Δz = z_true − target_z_mm.

    Requires ``ap_mass`` to be set on the parent SampleConfig.
    """
    eps2_values: List[float] = field(default_factory=list)
    z_branch: str = "true_ap.vtx_z_"
    betagamma_branch: str = "true_ap_betagamma"
    gen_length_mm: float = 1500.0
    target_z_mm: float = -1.1
    base_weight: str = "weight"


@dataclass
class WeightScanEntry:
    """One entry in a per-sample weight scan (label + weight expression).

    Used by :attr:`SampleConfig.weight_scan` to describe multiple logically
    identical samples that differ only in their event weight.  The sample's
    ROOT files are loaded once; the weight array is re-evaluated cheaply for
    each entry without touching disk again.
    """
    label: str
    weight: str
    color: str = ""                  # matplotlib color; leave empty to auto-assign from cycle
    epsilon_sq: Optional[float] = None  # ε² for Eq. 4 signal scaling in ABCD


@dataclass
class SimpConfig:
    """Dark sector parameters for SIMP signal MC reweighting.

    Used when ``SampleConfig.signal_type == "simp"``.  Branch names and
    generator parameters (z_branch, betagamma_branch, gen_length_mm,
    target_z_mm, base_weight) are taken from the sample's ``eps2_scan``
    block so there is no duplication.
    """
    alpha_d: float = 0.01           # dark sector coupling constant α_D
    vd_mass_ratio: float = 0.6      # m_V_D / m_A'  (benchmark: 0.6)
    pid_mass_ratio: float = 0.3333  # m_π_D / m_A'  (benchmark: 1/3)


@dataclass
class SampleConfig:
    name: str
    directory: Union[str, List[str]] = ""
    sample_type: str = "background"  # "data", "background", "signal"
    label: str = ""
    color: str = "black"
    weight: str = "1.0"
    scale: float = 1.0
    lumi_scale: bool = False  # if True, compute weight from summary.json
    tree: str = ""  # per-sample override
    selection: str = ""  # per-sample selection applied above region cuts
    rad_frac_role: str = ""  # "numerator" or "denominator" for radiative fraction
    aliases: Dict[str, str] = field(default_factory=dict)  # per-sample alias overrides (merged over global aliases)
    derive_smearing: bool = True  # if False, plot resolution but skip smearing derivation for this sample
    show_scale: bool = False      # if True, include this sample in the scale correction panel
    ap_mass: Optional[float] = None    # A' pole mass [GeV]; triggers Eq. 4 signal scaling when set
    epsilon_sq: Optional[float] = None  # ε² coupling; required alongside ap_mass
    run_min: Optional[int] = None      # inclusive lower bound on run number (data only)
    run_max: Optional[int] = None      # inclusive upper bound on run number (data only)
    exclude_runs: List[int] = field(default_factory=list)  # run numbers to skip entirely
    # Multiple weight scenarios sharing the same ROOT data (load once, weight N times).
    # When non-empty, weight_scan_exprs() iterates over these instead of the default weight.
    weight_scan: List[WeightScanEntry] = field(default_factory=list)
    eps2_scan: Optional["Eps2ScanConfig"] = None
    signal_type: str = "aprime"          # "aprime" or "simp"
    simp: Optional["SimpConfig"] = None  # dark sector params; required when signal_type="simp"

    def __post_init__(self):
        # Normalize directory to a list
        if isinstance(self.directory, str):
            self.directories = [self.directory]
        else:
            self.directories = list(self.directory)
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
    is_scaling_region: bool = False  # if True, used as preselection for Eq. 4 signal scaling

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
    # Optional 2D fields — if y_variable is set, this is a 2D histogram
    y_variable: str = ""
    y_bins: int = 50
    y_min: float = 0.0
    y_max: float = 1.0
    y_label_2d: str = ""  # y-axis label for 2D (y_label is repurposed as "Events" for 1D)
    tool_variable_name: str = ""  # track variable name for TrackSmearingTool binned lookup (e.g. "tanLambda")
    y_top_scale: float = 1.0  # multiply auto top y-limit by this factor (e.g. 5.0 adds headroom for legend)


@dataclass
class FitConfig:
    function: str = "exponential"  # only "exponential" for now
    sample: Union[str, List[str]] = ""  # which sample(s) to fit
    x_min: float = None            # fit range (defaults to histogram range)
    x_max: float = None
    p0: List[float] = field(default_factory=list)  # initial parameters [A, lambda]
    color: str = "red"
    show_params: bool = True       # display fit result text on plot

    def __post_init__(self):
        # Normalize sample to a list
        if isinstance(self.sample, str):
            self.samples = [self.sample] if self.sample else []
        else:
            self.samples = list(self.sample)


@dataclass
class SmearingConfig:
    data_sample: str
    mc_sample: str
    smearing_type: str = "absolute"  # "absolute" or "relative"
    fit_range: List[float] = field(default_factory=list)  # [min, max] for Y-axis Gaussian fit
    sigma_range: float = 1.5         # iterative window half-width in sigmas (overridden when core_fraction is set)
    core_fraction: Optional[float] = None  # if set, fit only the central fraction (e.g. 0.70); seeds window from percentiles
    json_output: str = ""            # path for JSON output; empty = no JSON
    top_region: str = ""             # region name for top-volume tracks
    bot_region: str = ""             # region name for bot-volume tracks
    tool_json_output: str = ""       # path for TrackSmearingTool-compatible JSON
    tool_section: str = ""           # section name in tool JSON (e.g. "pSmearing", "omegaSmearing")
    extra_samples: List[str] = field(default_factory=list)  # additional samples overlaid on fit plots (no smearing derived)
    scale_panel: bool = True           # if False, suppress the scale correction panel entirely
    scale_reference: Optional[float] = None  # None → divide by beam_energy (show μ/E_beam, line at 1);
                                             # float → show μ directly, reference line at this value (e.g. 0.0 for z0)


@dataclass
class BinnedConfig:
    """Configuration for binned comparison plots (variable distribution in bins of bin_variable)."""
    variable: str               # expression for x-axis (e.g. alias "ele_z0")
    bin_variable: str           # expression to bin in (e.g. alias "ele_tanl")
    bin_edges: List[float]      # edges of the bins in bin_variable
    bins: int = 80              # number of histogram bins on x-axis
    x_min: float = -2.0
    x_max: float = 2.0
    x_label: str = ""
    bin_label: str = ""         # label for the bin variable shown in panel titles
    label: str = ""             # extra label shown on the plots (e.g. "Electron", "Positron")
    normalize: bool = True      # normalize each histogram to unit area
    log_y: bool = False
    ratio_y_min: float = 0.5
    ratio_y_max: float = 1.5
    selection: str = ""         # additional event selection applied before binning
    json_output: str = ""       # path to write run-by-run fit results JSON; empty = no output
    json_key_prefix: str = ""   # prefix for JSON keys, e.g. "ele" → keys "ele_top", "ele_bot"


@dataclass
class ANNConfig:
    """Configuration for ANN-based signal/background scoring.

    When set on the global Config, the ANN score is computed per-event and
    injected into the data dict under ``score_variable`` before region
    selections are evaluated.  Region selections can then reference
    ``ann_score > 0.9`` (or whichever name / threshold you choose).
    """
    model_path: str                    # path to PyTorch .pt state-dict file
    scaler_path: str                   # path to sklearn .pkl or numpy .npz scaler
    score_variable: str = "ann_score"  # variable name available in selections


@dataclass
class LumiProjectionConfig:
    """One luminosity projection for the ABCD validation study.

    Specify either ``directory`` (observed data available; lumi computed dynamically)
    or ``scale_factor`` (projection only; no data to compare against).
    """
    label: str
    color: str = "blue"
    linestyle: str = "--"
    # Option A — directory-based: load observed N_A and compute lumi from files
    directory: str = ""
    lumi_file: str = ""      # overrides global lumi_file for this projection
    # Option B — scale-factor-based: project without observed data
    scale_factor: float = 0.0  # multiply reference luminosity by this factor


@dataclass
class ABCDAuxConfig:
    """One aux-variable histogram produced alongside an ABCD plot.

    Regions in the parent plot are treated as base regions.  For each bin of
    *bins* the system synthesises a child RegionConfig that appends
    ``(variable >= lo) & (variable < hi)`` to the base selection, runs the
    ABCD machinery, and integrates the result into a single histogram bin.
    """
    variable: str                           # expression for the aux variable (e.g. "psum")
    bins: List[float] = field(default_factory=list)  # bin edges (N+1 values for N bins)
    label: str = ""                         # x-axis label for the histogram


@dataclass
class ABCDConfig:
    mass_variable: str                      # expression for invariant mass
    z_variable: str                         # expression for vertex Z position
    mass_scan_min: float = 0.03             # GeV: start of mass scan
    mass_scan_max: float = 0.20             # GeV: end of mass scan (inclusive)
    mass_scan_step: float = 0.005           # GeV: step between mass bin centers
    mass_window_half_width: float = 0.005   # GeV: signal window = center ± half_width
    sideband_gap: float = 0.002             # GeV: gap from signal window edge to sideband
    sideband_width: float = 0.010           # GeV: width of each sideband (low + high)
    z_signal_min: float = 10.0              # mm: Z signal region lower bound
    z_signal_max: float = 100.0             # mm: Z signal region upper bound
    z_control_min: float = 2.0              # mm: Z control region lower bound
    z_control_max: float = 5.0              # mm: Z control region upper bound
    x_label: str = "$m_{e^+e^-}$ [GeV]"    # mass axis label
    log_y: bool = True                      # log scale on the y-axis
    show_mc_components: bool = False        # if True, draw individual MC process lines
    # --- mass-dependent window sizing ---
    # When mass_resolution contains an entry for the current region, the fixed
    # mass_window_half_width / sideband_gap / sideband_width fields are replaced by:
    #   SR half-width  = sr_sigmas  * sigma(m)
    #   gap            = gap_sigmas * sigma(m)
    #   sideband width = sb_sigmas  * sigma(m)
    # where sigma(m) is evaluated from the expression string (variable: m in GeV).
    sr_sigmas: float = 1.5                  # signal region half-width in units of sigma
    gap_sigmas: float = 0.5                 # gap between SR edge and sideband, in sigma
    sb_sigmas: float = 3.0                  # sideband width in units of sigma
    mass_resolution: Dict[str, str] = field(default_factory=dict)  # region → sigma(m) expression
    lumi_projections: List[LumiProjectionConfig] = field(default_factory=list)
    json_output: str = ""  # path to write per-mass-bin results JSON; empty = no output
    # --- signal normalization region ---
    # Name of a region (defined in the global regions list) to use as the
    # reference for Eq. 4 signal normalization.  N_data and N_mc are both
    # counted in this region (mass window applied, no z cut).  When empty,
    # the ABCD region itself is used (default / tighter behaviour).
    signal_norm_region: str = ""
    # --- mass window size systematic scan ---
    # List of signal-window half-widths [GeV] to evaluate alongside the nominal.
    # For each hw value the signal yield is recomputed using in-memory arrays
    # (no extra ROOT I/O).  Results are stored in the JSON and a comparison plot
    # is produced showing nsig vs hw for every signal sample on one figure.
    # Leave empty (default) to disable the scan entirely.
    window_scan: List[float] = field(default_factory=list)
    # --- 2D signal yield plot (nsig vs mass and ε²) ---
    # When non-empty, produce a 2D heat map of nsig(mass, ε²) after the scan.
    # The value is used as a flag; output filename is auto-generated from region
    # name and output_format.
    signal_2d_output: str = ""
    # --- mass scan points ---
    # When True, mass centers are the sorted ap_mass values of signal samples
    # in the plot (filtered to [mass_scan_min, mass_scan_max]) rather than the
    # regular arange grid.  mass_scan_min/max are still used as bounds.
    mass_scan_from_signal_samples: bool = False
    # --- signal normalization window ---
    # When set, n_data and n_mc used in the Eq. 4 prompt-yield calculation are
    # counted in [m ± signal_norm_hw] rather than the ABCD signal-region window.
    # n_mc_a (the signal acceptance in region A) still uses the ABCD window.
    # Set to None (default) to use the ABCD window for both (original behaviour).
    signal_norm_hw: Optional[float] = None
    # --- aux variable histograms ---
    # Each entry auto-generates disjoint per-bin regions from the base regions in
    # the plot and produces a histogram of integrated ABCD / data / MC vs that variable.
    aux_histograms: List["ABCDAuxConfig"] = field(default_factory=list)


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
    output_dir: str = ""  # per-plot override; falls back to global output_dir
    ratio_y_min: float = 0.5
    ratio_y_max: float = 1.5
    fit: Optional[FitConfig] = None
    smearing: Optional[SmearingConfig] = None
    abcd: Optional[ABCDConfig] = None
    binned: Optional[BinnedConfig] = None

    def __post_init__(self):
        if self.plot_type not in ("stack", "overlay", "rad_frac", "smearing", "abcd", "binned", "run_trend"):
            raise ValueError(
                f"Invalid plot_type '{self.plot_type}' for plot '{self.name}'. "
                "Must be 'stack', 'overlay', 'rad_frac', 'smearing', 'abcd', 'binned', or 'run_trend'."
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
    beam_energy: Optional[float] = None  # beam energy in GeV; used for scale correction panel
    scaling_mass_variable: str = ""      # expression for invariant mass used in Eq. 4 scaling
    scaling_mass_window: float = 0.005   # half-width [GeV] of mass window for counting data events
    scaling_rad_frac: Union[float, str] = 0.05  # f_rad: float or mass-dependent expression in m [GeV] (Eq. 4)
    ann: Optional["ANNConfig"] = None    # ANN scorer config; if set, ann_score available in selections
    run_label: str = ""                  # set automatically by --per-file; used to key JSON outputs
    hit_cat_file: str = ""               # path to hit-category fraction table (run L1L1 L1L2 L2L1 L2L2 Other)
    hit_cat_ref_run: int = 0             # reference run for hit-category SF (SF = frac(run)/frac(ref))
    hit_cat_poly_order: int = -1         # degree of polynomial SF fit; -1 = use closest-run lookup
    target_lumi: Optional[float] = None  # if set, produce a second set of plots scaled to this luminosity [pb^-1]


def load_config(path: str) -> Config:
    """Load a YAML configuration file and return a Config object."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    ann_raw = raw.get("ann", None)
    config = Config(
        output_dir=raw.get("output_dir", "plots/"),
        output_format=raw.get("output_format", "pdf"),
        tree=raw.get("tree", ""),
        luminosity=raw.get("luminosity", 1.0),
        lumi_file=raw.get("lumi_file", ""),
        aliases=raw.get("aliases", {}),
        beam_energy=raw.get("beam_energy", None),
        scaling_mass_variable=raw.get("scaling_mass_variable", ""),
        scaling_mass_window=raw.get("scaling_mass_window", 0.005),
        scaling_rad_frac=raw.get("scaling_rad_frac", 0.05),
        ann=ANNConfig(**ann_raw) if ann_raw is not None else None,
        hit_cat_file=raw.get("hit_cat_file", ""),
        hit_cat_ref_run=raw.get("hit_cat_ref_run", 0),
        hit_cat_poly_order=raw.get("hit_cat_poly_order", -1),
        target_lumi=raw.get("target_lumi", None),
    )

    for s in raw.get("samples", []):
        s = dict(s)
        weight_scan_raw = s.pop("weight_scan", [])
        eps2_scan_raw   = s.pop("eps2_scan", None)
        simp_raw        = s.pop("simp", None)
        sample = SampleConfig(**s)
        sample.weight_scan = [WeightScanEntry(**e) for e in weight_scan_raw]
        if eps2_scan_raw is not None:
            sample.eps2_scan = Eps2ScanConfig(**eps2_scan_raw)
        if simp_raw is not None:
            sample.simp = SimpConfig(**simp_raw)
        if not sample.tree:
            sample.tree = config.tree
        config.samples.append(sample)

    for r in raw.get("regions", []):
        config.regions.append(RegionConfig(**r))

    for h in raw.get("histograms", []):
        config.histograms.append(HistogramConfig(**h))

    for p in raw.get("plots", []):
        fit_raw     = p.pop("fit", None)
        smearing_raw = p.pop("smearing", None)
        abcd_raw    = p.pop("abcd", None)
        binned_raw  = p.pop("binned", None)
        plot = PlotConfig(**p)
        if fit_raw is not None:
            plot.fit = FitConfig(**fit_raw)
        if smearing_raw is not None:
            plot.smearing = SmearingConfig(**smearing_raw)
        if binned_raw is not None:
            plot.binned = BinnedConfig(**binned_raw)
        if abcd_raw is not None:
            lumi_proj_raw = abcd_raw.pop("lumi_projections", [])
            aux_hist_raw  = abcd_raw.pop("aux_histograms", [])
            plot.abcd = ABCDConfig(**abcd_raw)
            for lp in lumi_proj_raw:
                plot.abcd.lumi_projections.append(LumiProjectionConfig(**lp))
            for ah in aux_hist_raw:
                plot.abcd.aux_histograms.append(ABCDAuxConfig(**ah))
        config.plots.append(plot)

    # Auto-populate samples for rad_frac plots from rad_frac_role annotations
    rad_frac_samples = [s.name for s in config.samples if s.rad_frac_role]
    for plot in config.plots:
        if plot.plot_type == "rad_frac" and not plot.samples:
            plot.samples = list(rad_frac_samples)

    return config
