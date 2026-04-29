#!/usr/bin/env python3
"""
signal_rates.py

Computes three signal quantities prior to tight selection:

  1. aprime_rate  : A' production rate
                   = N_b_massbin * scale_const * ratio(mass) * mass * eps^2

  2. sarahs_stuff : aprime_rate * A_rad
                   Since ratio = f_rad / A_rad, this is:
                   = N_b_massbin * scale_const * f_rad * mass * eps^2

  3. simp_rate    : A' rate weighted by branching fractions and MC
                   acceptance integral (getSum equivalent)
                   = aprime_rate * (prho * rho_frac + pphi * phi_frac)

Usage (single point):
  python signal_rates.py \
      --base-module decayLength8sel \
      --eps 1e-4 --mass 100.0 --bg 500 \
      [--plots]

--bg is the number of background events already in the mass window.
It is used directly for the single-point calculation.
The plots use the mass-polynomial from plot_production_yields.py to
derive N_b_massbin per cell, but call the same compute_signal_rates()
function — so the physics is identical.

All physics code is taken verbatim from plot_production_yields.py.
The only new code is:
  - compute_signal_rates(): wraps the existing calculations, takes
    N_b_massbin as a plain number so the caller controls it
  - sarahs_stuff: f_rad numerator extracted from ratio()
  - make_plots(): adds a third panel and calls compute_signal_rates()
    per cell with N_b_massbin(mass) from the polynomial
"""

import sys
import math
import argparse
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Constants — verbatim from plot_production_yields.py
# ---------------------------------------------------------------------------
ALPHA_QED = 1.0 / 137.0459991
HBAR_C    = 1.973e-14   # GeV·cm
ALPHA_D   = 0.01

def scale_const():
    return 3.0 * math.pi / (2.0 * 1.0 * ALPHA_QED)

# ---------------------------------------------------------------------------
# Mass-window background polynomial — verbatim from plot_production_yields.py
# Used ONLY for the plots (to get N_b_massbin per mass cell).
# ---------------------------------------------------------------------------
def N_b_massbin(mass_mev):
    x = mass_mev / 1000.0
    poly = (-6860.03
            + 299358.0   * x
            - 4087220.0  * x**2
            + 25209900.0 * x**3
            - 73485900.0 * x**4
            + 82579800.0 * x**5)
    return 3.0e9 * poly / (82.9268041667 * 1000.0)

# ---------------------------------------------------------------------------
# Decay width helpers — verbatim from plot_production_yields.py
# ---------------------------------------------------------------------------
def beta_func(x, y):
    return (1 + y**2 - x**2 - 2*y) * (1 + y**2 - x**2 + 2*y)

def width_Ap_to_charged(alpha_D, f_pi_D, m_pi_D, m_V_D, m_Ap):
    x  = m_pi_D / m_Ap
    y  = m_V_D  / m_Ap
    Tv = 18.0 - (3.0/2.0 + 3.0/4.0)
    coeff = alpha_D * Tv / (192.0 * math.pi**4)
    return (coeff
            * (m_Ap / m_pi_D)**2
            * (m_V_D / m_pi_D)**2
            * (m_pi_D / f_pi_D)**4
            * m_Ap
            * beta_func(x, y)**1.5)

def width_Ap_to_vector(alpha_D, f_pi_D, m_pi_D, m_V_D, m_Ap, multiplicity=1.0):
    x           = m_V_D  / m_Ap
    y           = m_pi_D / m_Ap
    prefactor   = (alpha_D * multiplicity) / (192.0 * math.pi**4)
    ratio_terms = (m_Ap / m_pi_D)**2 * (m_V_D / m_pi_D)**2 * (m_pi_D / f_pi_D)**4
    return prefactor * ratio_terms * m_Ap * beta_func(x, y)**1.5

def width_Ap_to_invis(alpha_D, m_pi_D, m_V_D, m_Ap):
    term1 = 1 - (4.0 * m_pi_D**2) / m_Ap**2
    term2 = (m_V_D**2 / (m_Ap**2 - m_V_D**2))**2
    return (2.0 * alpha_D / 3.0) * m_Ap * term1**1.5 * term2

def rate_2l(alpha_D, f_pi_D, m_pi_D, m_V_D, m_Ap, epsilon, m_l, rho):
    alpha    = 1.0 / 137.0
    coeff    = (16 * math.pi * alpha_D * alpha * epsilon**2 * f_pi_D**2) / (3 * m_V_D**2)
    term1    = (m_V_D**2 / (m_Ap**2 - m_V_D**2))**2
    term2    = (1 - (4 * m_l**2 / m_V_D**2))**0.5
    term3    = 1 + (2 * m_l**2 / m_V_D**2)
    constant = 2 if rho else 1
    return coeff * term1 * term2 * term3 * m_V_D * constant

def dark_masses(mass_mev):
    m_pi_D = mass_mev / 3.0
    m_V_D  = 1.8 * mass_mev / 3.0
    f_pi_D = (mass_mev / 3.0) * (1.0 / (4.0 * math.pi))
    return m_pi_D, m_V_D, f_pi_D

# ---------------------------------------------------------------------------
# compute_signal_rates — the single method that returns all three quantities.
#
# N_b_massbin_val is passed in directly as a number.
#   - For single-point CLI use: pass --bg (background in the mass window).
#   - For plots: pass N_b_massbin(mass) from the polynomial above.
# This keeps the calculation identical in both cases.
# ---------------------------------------------------------------------------
def compute_signal_rates(base, mass_mev, epsilon, N_b_massbin_val):
    """
    Returns (aprime_rate, sarahs_stuff, simp_rate).

    base            : imported decayLength8sel module (needed for MC integral)
    mass_mev        : A' mass in MeV
    epsilon         : kinematic mixing parameter (not epsilon^2)
    N_b_massbin_val : background events in the mass window (plain number)
    """
    m_pi_D, m_V_D, f_pi_D = dark_masses(mass_mev)
    sc = scale_const()

    # --- ratio = f_rad / A_rad (verbatim from plot_production_yields / decayLength8sel) ---
    try:
        ratio_val = base.ratio(mass_mev)
    except Exception:
        ratio_val = 1.0

    # --- 1. A' production rate (ap_prod_yield from plot_production_yields.py) ---
    aprime_rate = N_b_massbin_val * sc * ratio_val * mass_mev * epsilon**2

    # --- 2. Sarah's stuff = aprime_rate * A_rad ---
    # ratio = f_rad / A_rad  =>  aprime_rate * A_rad = N_b * sc * f_rad * mass * eps^2
    # Extract f_rad (numerator of ratio) directly.
    x = mass_mev / 1000.0
    if 0.05 <= x <= 0.25:
        f_rad = (-.16647 + 8.0747*x - 111.31*x*x + 727.92*x**3
                 - 2241.3*x**4 + 2604.3*x**5)
    else:
        f_rad = 0.0
    sarahs_stuff = N_b_massbin_val * sc * f_rad * mass_mev * epsilon**2

    # --- branching fractions — verbatim from plot_production_yields.py ---
    rho_w_br  = width_Ap_to_vector(ALPHA_D, f_pi_D, m_pi_D, m_V_D, mass_mev, multiplicity=0.75)
    phi_w_br  = width_Ap_to_vector(ALPHA_D, f_pi_D, m_pi_D, m_V_D, mass_mev, multiplicity=1.5)
    invis_w   = width_Ap_to_invis(ALPHA_D, m_pi_D, m_V_D, mass_mev)
    charged_w = width_Ap_to_charged(ALPHA_D, f_pi_D, m_pi_D, m_V_D, mass_mev)
    total_w   = rho_w_br + phi_w_br + invis_w + charged_w
    rho_frac  = rho_w_br / total_w
    phi_frac  = phi_w_br / total_w

    # --- decay lengths — verbatim from plot_production_yields.py ---
    rho_width  = rate_2l(ALPHA_D, f_pi_D, m_pi_D, m_V_D, mass_mev, epsilon, 0.511, True)
    phi_width  = rate_2l(ALPHA_D, f_pi_D, m_pi_D, m_V_D, mass_mev, epsilon, 0.511, False)
    rho_length = (1000 * HBAR_C * 10.0) / rho_width
    phi_length = (1000 * HBAR_C * 10.0) / phi_width

    # --- MC acceptance integral — verbatim from plot_production_yields.py main loop ---
    try:
        den_edges, den_vals   = base.read_den_hist(m_V_D)
        psum_edges, psum_vals = base.read_psum_hist(m_V_D)
        mkey      = base._mass_key(m_V_D)
        events    = base._events_cache(mkey)
        mask_hist = base.tight_selection(events, 0, 0)
        zvals     = events["true_vd.vtx_z_"][mask_hist]
        num_vals, _ = np.histogram(zvals, bins=den_edges)
    except Exception as e:
        sys.stderr.write(f"[warn] MC unavailable for mass {mass_mev} MeV: {e}\n")
        return aprime_rate, sarahs_stuff, np.nan

    prho = 0.0
    pphi = 0.0
    for I in range(len(den_vals)):
        Ngen   = float(den_vals[I]) or 1.0
        Nacc   = float(num_vals[I])
        z_cent = max(0.0, 0.5 * (den_edges[I] + den_edges[I+1]))
        for J in range(len(psum_vals)):
            psum_val = 0.5 * (psum_edges[J] + psum_edges[J+1])
            gamma    = 1000.0 * psum_val / m_V_D
            prho += psum_vals[J] * (Nacc/Ngen) * np.exp(-z_cent/(gamma*rho_length)) / (rho_length*gamma)
            pphi += psum_vals[J] * (Nacc/Ngen) * np.exp(-z_cent/(gamma*phi_length)) / (phi_length*gamma)

    # --- 3. SIMP rate (acc_yield from write_final_yields / plot_production_yields) ---
    simp_rate = aprime_rate * (prho * rho_frac + pphi * phi_frac)

    return aprime_rate, sarahs_stuff, simp_rate

# ---------------------------------------------------------------------------
# Plotting — verbatim _add_colz from plot_production_yields.py, extended to
# three panels. Per-cell N_b_massbin comes from the polynomial (plots only).
# ---------------------------------------------------------------------------
CONTOUR_LEVELS = [1e2, 1e3, 1e4, 1e5, 1e6]
CONTOUR_COLORS = ["white", "lightyellow", "yellow", "orange", "red"]
CONTOUR_LABELS = {math.log10(v): f"$10^{{{int(math.log10(v))}}}$"
                  for v in CONTOUR_LEVELS}

def _add_colz(ax, masses, epsilons, Z, title, cmap="viridis"):
    """Verbatim from plot_production_yields.py."""
    log10Z = np.where(Z > 0, np.log10(Z), np.nan)
    vmin = np.nanmin(log10Z)
    vmax = np.nanmax(log10Z)
    pcm = ax.pcolormesh(masses, epsilons, log10Z,
                        shading="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax)
    ax.set_yscale("log")
    ax.set_xlim(50, 240)
    ax.set_ylim(1e-5, 1e-2)
    ax.set_xlabel(r"$m_{A'}$ [MeV]", fontsize=13)
    ax.set_ylabel(r"$\epsilon$",      fontsize=13)
    ax.set_title(title,               fontsize=13)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True)
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(N_{\rm signal\ produced})$", fontsize=11)
    log_levels = [math.log10(v) for v in CONTOUR_LEVELS
                  if vmin < math.log10(v) < vmax]
    if log_levels:
        cs = ax.contour(masses, epsilons, log10Z,
                        levels=log_levels,
                        colors=CONTOUR_COLORS[:len(log_levels)],
                        linewidths=1.5)
        ax.clabel(cs, fmt={lv: CONTOUR_LABELS[lv] for lv in log_levels},
                  fontsize=10, inline=True, inline_spacing=8)
    return pcm

def make_plots(base, masses, epsilons, outfile):
    """
    Three-panel plot: A' rate, Sarah's stuff, SIMP rate.
    N_b_massbin is derived from the polynomial per cell (plots only).
    Calls compute_signal_rates() identically to the single-point path.
    """
    Z_ap    = np.full((len(epsilons), len(masses)), np.nan)
    Z_sarah = np.full((len(epsilons), len(masses)), np.nan)
    Z_simp  = np.full((len(epsilons), len(masses)), np.nan)

    for i_m, mass in enumerate(masses):
        print(f"  mass {mass:.1f} MeV  ({i_m+1}/{len(masses)})", flush=True)
        nb = N_b_massbin(mass)   # polynomial — plots only
        if nb <= 0:
            continue
        for i_e, eps in enumerate(epsilons):
            ap, ss, sr = compute_signal_rates(base, mass, eps, nb)
            Z_ap[i_e, i_m]    = ap
            Z_sarah[i_e, i_m] = ss
            Z_simp[i_e, i_m]  = sr

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    _add_colz(axes[0], masses, epsilons, Z_ap,
              r"$A'$ Production Yield (no cuts)", cmap="viridis")
    _add_colz(axes[1], masses, epsilons, Z_sarah,
              r"Sarah's Stuff ($A'$ rate $\times A_{\rm rad}$)", cmap="magma")
    _add_colz(axes[2], masses, epsilons, Z_simp,
              r"SIMP Production Yield (no cuts)", cmap="plasma")
    fig.suptitle(r"Signal Produced in Mass Window (prior to tight selection)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute A', Sarah's stuff, and SIMP signal rates prior to tight selection.")
    ap.add_argument("--base-module", default="decayLength8sel",
                    help="Signal base module (same as in write_final_yields).")
    ap.add_argument("--eps",  type=float, required=True,
                    help="Epsilon (kinematic mixing parameter, not eps^2).")
    ap.add_argument("--mass", type=float, required=True,
                    help="A' mass in MeV.")
    ap.add_argument("--bg",   type=float, required=True,
                    help="Background events already in the mass window.")
    ap.add_argument("--plots", action="store_true",
                    help="Also produce three-panel colour plot over (mass, epsilon) grid.")
    ap.add_argument("--masses", nargs="+", type=float, default=None,
                    help="Mass grid for plots [MeV]. Default: linspace(50,240,40).")
    ap.add_argument("--mass-file", type=str, default=None,
                    help="Text file with one mass per line for plot grid.")
    ap.add_argument("--n-eps", type=int, default=80,
                    help="Number of epsilon points for plot grid.")
    ap.add_argument("--outfile", default="prod_yields.png",
                    help="Output PNG for plots.")
    args = ap.parse_args()

    try:
        base = importlib.import_module(args.base_module)
    except Exception as e:
        sys.exit(f"[error] Cannot import base module '{args.base_module}': {e}")

    # --- single-point calculation: use --bg directly, no polynomial ---
    aprime_rate, sarahs_stuff, simp_rate = compute_signal_rates(
        base, args.mass, args.eps, args.bg)

    print("=" * 55)
    print(f"  mass (MeV)               = {args.mass:.3f}")
    print(f"  epsilon                  = {args.eps:.4e}")
    print(f"  bg in mass window        = {args.bg:.4e}")
    print("-" * 55)
    print(f"  A' production rate       = {aprime_rate:.6e}")
    print(f"  Sarah's stuff (x A_rad)  = {sarahs_stuff:.6e}")
    print(f"  SIMP production rate     = {simp_rate:.6e}")
    print("=" * 55)

    if args.plots:
        if args.mass_file:
            plot_masses = np.loadtxt(args.mass_file)
        elif args.masses:
            plot_masses = np.array(sorted(args.masses))
        else:
            plot_masses = np.linspace(50, 240, 40)
        plot_epsilons = np.logspace(-5, -2, args.n_eps)
        print(f"Grid: {len(plot_masses)} masses x {len(plot_epsilons)} epsilons")
        make_plots(base, plot_masses, plot_epsilons, args.outfile)


if __name__ == "__main__":
    main()
