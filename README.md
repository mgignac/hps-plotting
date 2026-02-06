# hpsplot

Plotting tools for HPS (Heavy Photon Search) analysis. Reads ROOT ntuples produced by `hpstr`, applies selections, fills histograms, and generates publication-quality matplotlib plots — all driven by a single YAML config file.

## Installation

```bash
cd /sdf/data/hps/users/mgignac/software/2021-ana/plotting
pip install -e .
```

**Dependencies:** numpy, matplotlib, uproot, awkward, pyyaml

## Quick Start

```bash
python -m hpsplot examples/config.yaml -v
```

This reads the YAML config, loads ROOT files, applies selections, fills histograms, and saves plots to the configured output directory.

### CLI Options

```
python -m hpsplot <config.yaml> [-o OUTPUT_DIR] [-v]

positional arguments:
  config                Path to YAML configuration file

options:
  -o, --output-dir      Override output directory
  -v, --verbose         Enable verbose (DEBUG) logging
```

## YAML Configuration

A config file defines four sections: **samples**, **regions**, **histograms**, and **plots**.

### Samples

Each sample points to a directory of ROOT files (or a single file) and specifies how to treat it.

```yaml
tree: "vtxana_kf_Tight_tree"    # default TTree name

samples:
  - name: data
    directory: /path/to/data/file.root
    sample_type: data            # data | background | signal
    label: "Data"
    color: "black"
    weight: "1.0"                # expression for per-event weight
    scale: 1.0                   # global scale factor

  - name: tridents
    directory: /path/to/tridents/   # all *.root files are chained
    sample_type: background
    label: "Tridents"
    color: "#e74c3c"
    weight: "weight"             # reads the "weight" branch
    scale: 42.6
```

- `directory` — path to a single `.root` file or a directory (chains all `*.root` files via `uproot.concatenate`)
- `sample_type` — controls plotting behavior: `data` is drawn as points, `background` is stacked, `signal` is overlaid as dashed lines
- `weight` — any valid expression using branch names (e.g., `"weight * sf"`)
- `scale` — multiplied on top of per-event weights (useful for cross-section normalization)

### Regions

Regions define event selections using expressions evaluated on branch arrays.

```yaml
regions:
  - name: tight
    selection: "ele_p > 0.4 and pos_p > 0.4 and vtx_chi2 < 10"
    label: "Tight"

  - name: preselection
    selection: ""      # empty = no cut (all events pass)
    label: "Preselection"
```

### Histograms

Define what to plot and how to bin it.

```yaml
histograms:
  - name: vtx_mass
    variable: vtx_invM_          # expression to histogram
    bins: 50
    x_min: 0.0
    x_max: 0.2
    x_label: "$m_{vtx}$ [GeV]"
    y_label: "Events"            # optional, defaults to "Events"
    log_y: false                 # optional, defaults to false
```

The `variable` field supports any expression the parser understands (see below).

### Plots

Plots combine samples, regions, and histograms into output figures.

```yaml
plots:
  # Stacked MC backgrounds + data points + ratio panel
  - name: stack_tight
    plot_type: stack
    histograms: [vtx_mass, vtx_z]
    regions: [tight]
    samples: [tridents, wab, data]
    data_sample: data
    signal_samples: []           # optional: overlaid as dashed lines
    ratio_y_min: 0.5             # optional, defaults to 0.5
    ratio_y_max: 1.5             # optional, defaults to 1.5

  # Normalized shape comparison
  - name: overlay_comparison
    plot_type: overlay
    histograms: [vtx_mass]
    regions: [tight, loose]
    samples: [tridents, wab]
    normalize: true              # normalize each sample to unit area
```

**Plot types:**

| Type | Description |
|------|-------------|
| `stack` | Stacked MC backgrounds (filled), data points (markers with error bars), optional signal overlay (dashed), MC stat uncertainty band (hatched), and a data/MC ratio panel |
| `overlay` | Step histograms for each sample overlaid on the same axes, optionally normalized to unit area |

### Global Settings

```yaml
output_dir: "plots/"       # where to save figures
output_format: "pdf"       # pdf, png, svg, etc.
tree: "vtxana_kf_Tight_tree"  # default TTree name (can be overridden per sample)
```

## Expression Language

The `selection`, `variable`, and `weight` fields all use a safe expression parser (no `eval()`). Supported syntax:

| Feature | Examples |
|---------|----------|
| Branch names | `ele_p`, `vtx_chi2`, `vtx_invM_` |
| Arithmetic | `ele_p + pos_p`, `weight * 0.5`, `x / y` |
| Exponentiation | `ele_p ** 2` |
| Unary minus | `-vtx_Z` |
| Comparisons | `ele_p > 0.5`, `vtx_chi2 <= 10`, `n == 1`, `flag != 0` |
| Boolean logic | `ele_p > 0.4 and pos_p > 0.4`, `a > 1 or b < 2`, `not flag > 0` |
| Functions | `abs(vtx_Z)`, `sqrt(x)`, `log(y)`, `log10(z)`, `exp(w)` |
| Multi-arg functions | `max(ele_p, pos_p)`, `min(a, b)`, `pow(x, 2)` |
| Trig functions | `sin(x)`, `cos(x)`, `tan(x)` |
| Parentheses | `(ele_p + pos_p) * 2` |
| Constants | `1.0`, `3.14`, `1e-3` |

## Package Structure

```
plotting/
├── pyproject.toml
├── README.md
├── examples/
│   └── config.yaml
└── src/
    └── hpsplot/
        ├── __init__.py
        ├── __main__.py          # CLI entry point
        ├── config.py            # YAML → dataclasses
        ├── utils.py             # Expression tokenizer + recursive descent parser
        ├── sample.py            # ROOT file reading (uproot.concatenate)
        ├── region.py            # Selection mask evaluation
        ├── histogram.py         # Histogram filling + sqrt(sum(w²)) errors
        ├── results.py           # Processing loop: samples × regions × histograms
        └── plotting/
            ├── __init__.py
            ├── style.py         # HPS matplotlib style
            ├── stack.py         # Stacked MC + data + ratio panel
            └── overlay.py       # Normalized shape comparisons
```

## Processing Pipeline

1. Load YAML config into dataclasses
2. For each sample, determine needed branches from histogram variables, region selections, and weight expressions
3. Read ROOT files with `uproot.concatenate()`
4. For each region, compute a boolean selection mask
5. For each histogram, evaluate the variable and weight expressions on masked data
6. Fill `numpy.histogram` with `sqrt(sum(w²))` error propagation
7. Generate plots with the HPS matplotlib style
