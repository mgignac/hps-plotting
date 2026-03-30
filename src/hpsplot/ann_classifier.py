"""ANN classifier for signal/background discrimination.

Architecture (inferred from classifier_adv_2021_v9_pass5_run42QualCuts.pt):
    Input: 34 features
    Hidden: Linear(34→264) + BN + LeakyReLU  ×2
            Linear(264→128) + BN + LeakyReLU  ×2
            Linear(128→64)  + BN + LeakyReLU
    Output: Linear(64→1)  (sigmoid applied during inference)

Feature order (ANN-notebook convention):
    vertex pos x/y/z, psum,
    ele track ×12, pos track ×12,
    vertex chi2, vtx_proj_sig ×3, L1 iso significance ×2
"""

import logging
from pathlib import Path

import numpy as np
import uproot
import awkward as ak

logger = logging.getLogger(__name__)

# The 12 per-track quantities, in order
_TRACK_KEYS = [
    "n_hits_", "d0_", "phi0_", "z0_", "tan_lambda_",
    "px_", "py_", "pz_", "chi2_",
    "x_at_ecal_", "y_at_ecal_", "z_at_ecal_",
]

# Internal alias names added to effective_aliases so sample.load() can
# load the TVector3 vertex position components.
# Values follow the sample.py tvec-alias convention: path ending in .fX/.fY/.fZ
ANN_VTX_ALIASES = {
    "_ann_vtx_x": "vertex./vertex.pos_.fX",
    "_ann_vtx_y": "vertex./vertex.pos_.fY",
    "_ann_vtx_z": "vertex./vertex.pos_.fZ",
}

# All remaining ANN branches (loadable directly by sample.load as "direct" branches)
ANN_DIRECT_BRANCHES = (
    ["psum"]
    + [f"ele.track_.{k}" for k in _TRACK_KEYS]
    + [f"pos.track_.{k}" for k in _TRACK_KEYS]
    + [
        "vertex.chi2_",
        "vtx_proj_sig", "vtx_proj_x_sig", "vtx_proj_y_sig",
        "ele_L1_iso_significance", "pos_L1_iso_significance",
    ]
)


def build_ann_matrix(data):
    """Build the (N, 34) float32 feature matrix from a data dict.

    Parameters
    ----------
    data : dict
        Branch/alias name → numpy array, as returned by Sample.load().
        Must contain all ANN_DIRECT_BRANCHES and the _ann_vtx_x/y/z keys.
    """
    cols = [
        data["_ann_vtx_x"],
        data["_ann_vtx_y"],
        data["_ann_vtx_z"],
        data["psum"],
    ]
    for k in _TRACK_KEYS:
        cols.append(data[f"ele.track_.{k}"])
    for k in _TRACK_KEYS:
        cols.append(data[f"pos.track_.{k}"])
    cols += [
        data["vertex.chi2_"],
        data["vtx_proj_sig"],
        data["vtx_proj_x_sig"],
        data["vtx_proj_y_sig"],
        data["ele_L1_iso_significance"],
        data["pos_L1_iso_significance"],
    ]
    return np.column_stack(cols).astype(np.float32)


def load_ann(model_path, scaler_path):
    """Load the ANN model and its feature scaler.

    Parameters
    ----------
    model_path : str or Path
        PyTorch state-dict file (.pt).
    scaler_path : str or Path
        sklearn StandardScaler pickle (.pkl) or numpy archive (.npz with
        keys ``mean`` and ``scale``).

    Returns
    -------
    (model, scaler_mean, scaler_scale)
        model        — ANNClassifier in eval mode (CPU)
        scaler_mean  — float32 numpy array, shape (34,)
        scaler_scale — float32 numpy array, shape (34,)
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for ANN scoring: pip install torch"
        ) from exc

    # --- model architecture ------------------------------------------------
    class _ANNClassifier(nn.Module):
        def __init__(self, in_features=34):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 264, bias=False),
                nn.BatchNorm1d(264), nn.LeakyReLU(),
                nn.Linear(264, 264, bias=False),
                nn.BatchNorm1d(264), nn.LeakyReLU(),
                nn.Linear(264, 128, bias=False),
                nn.BatchNorm1d(128), nn.LeakyReLU(),
                nn.Linear(128, 128, bias=False),
                nn.BatchNorm1d(128), nn.LeakyReLU(),
                nn.Linear(128, 64, bias=False),
                nn.BatchNorm1d(64), nn.LeakyReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = _ANNClassifier(in_features=34)
    state = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded ANN model from %s", model_path)

    # --- scaler ------------------------------------------------------------
    scaler_path = str(scaler_path)
    if scaler_path.endswith(".pkl"):
        try:
            import joblib
            scaler = joblib.load(scaler_path)
            mean  = scaler.mean_.astype(np.float32)
            scale = scaler.scale_.astype(np.float32)
        except ImportError as exc:
            raise ImportError(
                "joblib is required to load the .pkl scaler: pip install joblib"
            ) from exc
    else:
        arr   = np.load(scaler_path)
        mean  = arr["mean"].astype(np.float32)
        scale = arr["scale"].astype(np.float32)

    logger.info("Loaded ANN scaler from %s  (mean shape %s)", scaler_path, mean.shape)
    return model, mean, scale


def predict_scores(model, scaler_mean, scaler_scale, X, batch_size=100_000):
    """Run ANN inference on a pre-built feature matrix.

    Parameters
    ----------
    model : _ANNClassifier in eval mode
    scaler_mean, scaler_scale : float32 arrays, shape (34,)
    X : float32 array, shape (N, 34)
    batch_size : int

    Returns
    -------
    float32 array, shape (N,), values in [0, 1]
    """
    import torch

    n = X.shape[0]
    scores = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk  = np.nan_to_num(X[start:end], nan=0.0, posinf=0.0, neginf=0.0)
        scaled = (chunk - scaler_mean) / scaler_scale
        tensor = torch.from_numpy(scaled)
        with torch.no_grad():
            scores[start:end] = torch.sigmoid(model(tensor)).cpu().numpy().ravel()

    return scores


def score_sample(sample_cfg, tree, model, scaler_mean, scaler_scale):
    """Load ANN feature branches from ROOT files and return per-event scores.

    This performs its own uproot read independently of Sample.load(), using
    filter_name to safely handle dotted branch names and TVector3 components.
    Used as a fallback when the branch-injection path (ANN_VTX_ALIASES +
    ANN_DIRECT_BRANCHES) is not available.

    Parameters
    ----------
    sample_cfg : SampleConfig
    tree : str  — TTree name
    model, scaler_mean, scaler_scale — from load_ann()

    Returns
    -------
    float32 array, shape (N_total_events,)
    """
    from .sample import _resolve_files

    patterns = []
    for d in sample_cfg.directories:
        files = _resolve_files(d,
                               run_min=sample_cfg.run_min,
                               run_max=sample_cfg.run_max)
        patterns.extend(f"{f}:{tree}" for f in files)

    if not patterns:
        raise ValueError(f"ANN scorer: no files for sample '{sample_cfg.name}'")

    pattern = patterns if len(patterns) > 1 else patterns[0]

    # --- branches needed ---------------------------------------------------
    flat_branches = set(ANN_DIRECT_BRANCHES)
    vtxpos_parent = "vertex.pos_"  # TVector3 — read separately

    # Load flat branches (dotted names OK with filter_name)
    arrays_flat = uproot.concatenate(
        pattern,
        filter_name=lambda name: name in flat_branches,
        library="ak",
    )

    # Load TVector3 parent
    arrays_vtx = uproot.concatenate(
        pattern,
        filter_name=lambda name: (
            name == vtxpos_parent
            or name == vtxpos_parent + "."
            or name.startswith(vtxpos_parent + ".")
        ),
        library="ak",
    )

    # Extract TVector3 components
    fields = arrays_vtx.fields
    vtxpos_key = vtxpos_parent if vtxpos_parent in fields else next(
        (f for f in fields if vtxpos_parent in f), None
    )
    if vtxpos_key is None:
        raise KeyError(
            f"Cannot find '{vtxpos_parent}' in ANN data. Fields: {fields}"
        )

    vtxpos = arrays_vtx[vtxpos_key]

    # Build data dict matching build_ann_matrix expectations
    data = {
        "_ann_vtx_x": ak.to_numpy(vtxpos["fX"]),
        "_ann_vtx_y": ak.to_numpy(vtxpos["fY"]),
        "_ann_vtx_z": ak.to_numpy(vtxpos["fZ"]),
    }
    for b in ANN_DIRECT_BRANCHES:
        data[b] = ak.to_numpy(arrays_flat[b])

    X = build_ann_matrix(data)
    return predict_scores(model, scaler_mean, scaler_scale, X)
