"""SIMP dark-sector signal rate computation.

Computes per-event lifetime weights for SIMP MC events by replacing the
standard A' single-exponential with a combined ρ_D / φ_D exponential weighted
by their branching fractions.  The Eq. 4 normalization in abcd.py is unchanged;
the branching fractions and decay acceptance flow through the weight sum.

Physics from rory/signal_rates.py (Chiu, HPS SIMP analysis).
"""

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

ALPHA_EM      = 1.0 / 137.036
HBAR_C_MEV_MM = 1.973e-10   # ℏc [MeV·mm]
M_ELECTRON_MEV = 0.511


# ---------------------------------------------------------------------------
# Dark sector mass relations
# ---------------------------------------------------------------------------

def dark_masses(ap_mass_mev, vd_ratio=0.6, pid_ratio=1.0 / 3.0):
    """Return (m_pi_D, m_V_D, f_pi_D) in MeV for benchmark mass ratios.

    Parameters
    ----------
    ap_mass_mev : float
        A' mass [MeV].
    vd_ratio : float
        m_V_D / m_A' (default 0.6).
    pid_ratio : float
        m_π_D / m_A' (default 1/3).

    Returns
    -------
    (m_pi_D, m_V_D, f_pi_D) : tuple of float, all in MeV.
    """
    m_pi_D = pid_ratio * ap_mass_mev
    m_V_D  = vd_ratio  * ap_mass_mev
    f_pi_D = m_pi_D / (4.0 * math.pi)
    logger.debug(
        "dark_masses(m_A'=%.2f MeV): m_pi_D=%.3f, m_V_D=%.3f, f_pi_D=%.4g [MeV]",
        ap_mass_mev, m_pi_D, m_V_D, f_pi_D,
    )
    return m_pi_D, m_V_D, f_pi_D


# ---------------------------------------------------------------------------
# Decay width helpers (verbatim from rory/signal_rates.py)
# ---------------------------------------------------------------------------

def _beta_func(x, y):
    return (1 + y**2 - x**2 - 2*y) * (1 + y**2 - x**2 + 2*y)


def _width_Ap_to_vector(alpha_d, f_pi_D, m_pi_D, m_V_D, m_ap, multiplicity):
    """Partial width for A' → V_D (ρ_D or φ_D)."""
    x         = m_V_D  / m_ap
    y         = m_pi_D / m_ap
    prefactor = (alpha_d * multiplicity) / (192.0 * math.pi**4)
    ratio_terms = (m_ap / m_pi_D)**2 * (m_V_D / m_pi_D)**2 * (m_pi_D / f_pi_D)**4
    bf = _beta_func(x, y)
    if bf <= 0:
        return 0.0
    return prefactor * ratio_terms * m_ap * bf**1.5


def _width_Ap_to_invis(alpha_d, m_pi_D, m_V_D, m_ap):
    """Partial width for A' → invisible (dark pion pairs)."""
    if m_ap <= 2.0 * m_pi_D or m_ap <= m_V_D:
        return 0.0
    term1 = 1 - (4.0 * m_pi_D**2) / m_ap**2
    if term1 <= 0:
        return 0.0
    term2 = (m_V_D**2 / (m_ap**2 - m_V_D**2))**2
    return (2.0 * alpha_d / 3.0) * m_ap * term1**1.5 * term2


def _width_Ap_to_charged(alpha_d, f_pi_D, m_pi_D, m_V_D, m_ap):
    """Partial width for A' → charged dark pions."""
    x  = m_pi_D / m_ap
    y  = m_V_D  / m_ap
    Tv = 18.0 - (3.0 / 2.0 + 3.0 / 4.0)
    coeff = alpha_d * Tv / (192.0 * math.pi**4)
    bf = _beta_func(x, y)
    if bf <= 0:
        return 0.0
    return (coeff
            * (m_ap / m_pi_D)**2
            * (m_V_D / m_pi_D)**2
            * (m_pi_D / f_pi_D)**4
            * m_ap
            * bf**1.5)


def _rate_2l(alpha_d, f_pi_D, m_pi_D, m_V_D, m_ap, eps2, m_l_mev, rho):
    """Partial decay rate V_D → ℓ+ℓ- [MeV], via kinematic mixing."""
    if m_V_D <= 2.0 * m_l_mev:
        return 0.0
    alpha = ALPHA_EM
    coeff = (16 * math.pi * alpha_d * alpha * eps2 * f_pi_D**2) / (3 * m_V_D**2)
    term1 = (m_V_D**2 / (m_ap**2 - m_V_D**2))**2
    term2 = math.sqrt(1 - (4 * m_l_mev**2 / m_V_D**2))
    term3 = 1 + (2 * m_l_mev**2 / m_V_D**2)
    const = 2 if rho else 1
    return coeff * term1 * term2 * term3 * m_V_D * const


# ---------------------------------------------------------------------------
# Public physics API
# ---------------------------------------------------------------------------

def dark_branching_fractions(alpha_d, m_pi_D, m_V_D, f_pi_D, ap_mass_mev):
    """Compute BR(A' → ρ_D) and BR(A' → φ_D).

    Returns
    -------
    (rho_frac, phi_frac) : tuple of float
        Branching fractions into the visible dark-vector channels.
        rho_frac + phi_frac < 1 because invisible and charged modes also contribute.
    """
    rho_w   = _width_Ap_to_vector(alpha_d, f_pi_D, m_pi_D, m_V_D, ap_mass_mev, multiplicity=0.75)
    phi_w   = _width_Ap_to_vector(alpha_d, f_pi_D, m_pi_D, m_V_D, ap_mass_mev, multiplicity=1.5)
    invis_w = _width_Ap_to_invis(alpha_d, m_pi_D, m_V_D, ap_mass_mev)
    charg_w = _width_Ap_to_charged(alpha_d, f_pi_D, m_pi_D, m_V_D, ap_mass_mev)
    total_w = rho_w + phi_w + invis_w + charg_w

    if total_w <= 0:
        logger.warning(
            "dark_branching_fractions: total width = 0 at m_A'=%.2f MeV — returning (0, 0).",
            ap_mass_mev,
        )
        return 0.0, 0.0

    rho_frac = rho_w / total_w
    phi_frac = phi_w / total_w

    logger.debug(
        "dark_branching_fractions(m_A'=%.2f MeV, alpha_d=%.3g): "
        "Γ_ρ=%.4g  Γ_φ=%.4g  Γ_invis=%.4g  Γ_charg=%.4g  Γ_tot=%.4g  "
        "BR_ρ=%.4f  BR_φ=%.4f  BR_vis=%.4f",
        ap_mass_mev, alpha_d,
        rho_w, phi_w, invis_w, charg_w, total_w,
        rho_frac, phi_frac, rho_frac + phi_frac,
    )
    return rho_frac, phi_frac


def dark_vector_ctau(alpha_d, m_pi_D, m_V_D, f_pi_D, ap_mass_mev, eps2,
                     m_l_mev=M_ELECTRON_MEV):
    """Compute decay lengths cτ_ρ and cτ_φ [mm] for the dark vectors.

    Uses the partial width V_D → e+e- (kinematic mixing).

    Parameters
    ----------
    eps2 : float
        ε² kinematic mixing parameter.
    m_l_mev : float
        Lepton mass [MeV] (default: electron).

    Returns
    -------
    (ctau_rho_mm, ctau_phi_mm) : tuple of float
    """
    gamma_rho = _rate_2l(alpha_d, f_pi_D, m_pi_D, m_V_D, ap_mass_mev,
                         eps2, m_l_mev, rho=True)
    gamma_phi = _rate_2l(alpha_d, f_pi_D, m_pi_D, m_V_D, ap_mass_mev,
                         eps2, m_l_mev, rho=False)

    ctau_rho = (HBAR_C_MEV_MM / gamma_rho) if gamma_rho > 0 else float("inf")
    ctau_phi = (HBAR_C_MEV_MM / gamma_phi) if gamma_phi > 0 else float("inf")

    logger.debug(
        "dark_vector_ctau(eps2=%.3g, m_A'=%.2f MeV): "
        "Γ_ρ=%.4g MeV  cτ_ρ=%.4g mm  Γ_φ=%.4g MeV  cτ_φ=%.4g mm",
        eps2, ap_mass_mev, gamma_rho, ctau_rho, gamma_phi, ctau_phi,
    )
    return ctau_rho, ctau_phi


def simp_event_weights(z_arr, bg_arr, bw_arr, gen_length_mm,
                       ctau_rho, ctau_phi, rho_frac, phi_frac,
                       target_z_mm=-1.1):
    """Per-event SIMP weights combining ρ_D and φ_D lifetime PDFs.

    Replaces the single-exponential A' weight with:
        w = bw × gen_L × (rho_frac × lt_ρ + phi_frac × lt_φ)

    where lt_V = exp(-Δz / (βγ·cτ_V)) / (βγ·cτ_V) and Δz = z - target_z_mm.
    Events with Δz < 0 or non-finite/negative results are zeroed.

    Parameters
    ----------
    z_arr, bg_arr, bw_arr : array-like
        Dark-vector truth z [mm], βγ, and base event weights.
    gen_length_mm : float
        Generator decay length [mm] (same role as in A' eps2_scan).
    ctau_rho, ctau_phi : float
        Decay lengths [mm] for ρ_D and φ_D.
    rho_frac, phi_frac : float
        Branching fractions.
    target_z_mm : float
        Target z position [mm] (default -1.1 mm for HPS).

    Returns
    -------
    weights : np.ndarray
    """
    z_arr  = np.asarray(z_arr,  dtype=float)
    bg_arr = np.asarray(bg_arr, dtype=float)
    bw_arr = np.asarray(bw_arr, dtype=float)

    dz    = z_arr - target_z_mm
    L_rho = bg_arr * ctau_rho
    L_phi = bg_arr * ctau_phi

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lt_rho = np.where((dz >= 0) & (L_rho > 0), np.exp(-dz / L_rho) / L_rho, 0.0)
        lt_phi = np.where((dz >= 0) & (L_phi > 0), np.exp(-dz / L_phi) / L_phi, 0.0)

    w = bw_arr * gen_length_mm * (rho_frac * lt_rho + phi_frac * lt_phi)
    return np.where(np.isfinite(w) & (w >= 0), w, 0.0)


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_simp_diagnostics(simp_entries, ap_entries, region_name, output_dir, output_format):
    """Two-panel diagnostic: nsig vs ε² for SIMP and A', plus SIMP/A' ratio.

    Parameters
    ----------
    simp_entries : list of dict
        Entries from signal_scan_cache with "simp" in label.
        Each dict: {label, ap_mass, eps2, nsig (array), ...}
    ap_entries : list of dict or None
        Corresponding A' entries for comparison (may be None or empty).
    region_name : str
    output_dir : str or Path
    output_format : str
    """
    if not simp_entries:
        return

    has_ap = bool(ap_entries)

    # Log summary
    for e in simp_entries:
        logger.info(
            "SIMP diag [%s | %s | eps2=%.3g]: nsig_sum=%.4g",
            region_name, e.get("label", "?"), e.get("eps2", float("nan")),
            float(np.sum(e["nsig"])),
        )

    # Group by ap_mass
    simp_by_mass = {}
    for e in simp_entries:
        m = e.get("ap_mass")
        simp_by_mass.setdefault(m, []).append(e)

    ap_by_mass = {}
    for e in (ap_entries or []):
        m = e.get("ap_mass")
        ap_by_mass.setdefault(m, []).append(e)

    n_panels = 2 if has_ap else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(7, 4 * n_panels), squeeze=False)
    ax_nsig  = axes[0, 0]
    ax_ratio = axes[1, 0] if has_ap else None

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = iter(colors * 10)

    for mass, entries in sorted(simp_by_mass.items()):
        c    = next(color_cycle)
        eps2s = [e["eps2"] for e in sorted(entries, key=lambda x: x["eps2"])]
        nsigs = [float(np.sum(e["nsig"])) for e in sorted(entries, key=lambda x: x["eps2"])]
        label = f"SIMP m={mass*1000:.0f} MeV" if mass is not None else "SIMP"
        ax_nsig.plot(eps2s, nsigs, "o-", color=c, label=label, linewidth=1.5, markersize=4)

        if ax_ratio is not None and mass in ap_by_mass:
            ap_entries_m = sorted(ap_by_mass[mass], key=lambda x: x["eps2"])
            ap_eps2s = [e["eps2"] for e in ap_entries_m]
            ap_nsigs = [float(np.sum(e["nsig"])) for e in ap_entries_m]
            # Interpolate A' nsig at SIMP eps2 values for ratio
            ap_map = dict(zip(ap_eps2s, ap_nsigs))
            ratio_x, ratio_y = [], []
            for e2, ns in zip(eps2s, nsigs):
                if e2 in ap_map and ap_map[e2] > 0:
                    ratio_x.append(e2)
                    ratio_y.append(ns / ap_map[e2])
            if ratio_x:
                ax_ratio.plot(ratio_x, ratio_y, "o-", color=c,
                              label=f"m={mass*1000:.0f} MeV", linewidth=1.5, markersize=4)

    # Overlay A' curves (dashed)
    if has_ap:
        for mass, entries in sorted(ap_by_mass.items()):
            c    = next(color_cycle)
            eps2s = [e["eps2"] for e in sorted(entries, key=lambda x: x["eps2"])]
            nsigs = [float(np.sum(e["nsig"])) for e in sorted(entries, key=lambda x: x["eps2"])]
            ax_nsig.plot(eps2s, nsigs, "--", color=c,
                         label=f"A' m={mass*1000:.0f} MeV" if mass else "A'",
                         linewidth=1.2, alpha=0.7)

    ax_nsig.set_xscale("log")
    ax_nsig.set_yscale("log")
    ax_nsig.set_xlabel(r"$\varepsilon^2$", fontsize=11)
    ax_nsig.set_ylabel(r"$N_{\rm sig}$ (summed over mass bins)", fontsize=10)
    ax_nsig.set_title(f"SIMP vs A' expected yield — {region_name}", fontsize=11)
    ax_nsig.legend(fontsize=8, loc="upper left")
    ax_nsig.grid(True, which="both", alpha=0.3)

    if ax_ratio is not None:
        ax_ratio.set_xscale("log")
        ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_ratio.set_xlabel(r"$\varepsilon^2$", fontsize=11)
        ax_ratio.set_ylabel(r"$N_{\rm sig}^{\rm SIMP} / N_{\rm sig}^{A'}$", fontsize=10)
        ax_ratio.set_title("SIMP / A' yield ratio", fontsize=11)
        ax_ratio.legend(fontsize=8)
        ax_ratio.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"simp_diagnostics_{region_name}.{output_format}"
    fig.savefig(outdir / fname, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SIMP diagnostic plot: %s", outdir / fname)
