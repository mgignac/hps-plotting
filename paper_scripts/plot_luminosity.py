import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

df = pd.read_csv("HPS_Runs_2021.csv")
df["end_time"] = pd.to_datetime(df["end_time"])
df = df.sort_values("end_time").reset_index(drop=True)

# Split into main and Moller runs
is_moller = df["run_config"].fillna("").str.startswith("hps_1.9_")
selected = df["selected"] == True
main = df[~is_moller & selected].copy()
moller = df[is_moller & selected].copy()

# Cumulative sum of luminosity (column N = "luminosity")
main["cum_lumi"] = main["luminosity"].cumsum()
moller["cum_lumi"] = moller["luminosity"].cumsum()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for ax, data, label, color, day_interval in [
    (axes[0], main,   "Main Production Runs (3.74 GeV)", "steelblue", 3),
    (axes[1], moller, "Møller Runs (1.92 GeV)",          "firebrick", 1),
]:
    ax.step(data["end_time"], data["cum_lumi"], where="post", color=color, linewidth=1.8)
    ax.fill_between(data["end_time"], data["cum_lumi"], step="post", alpha=0.15, color=color)
    ax.set_title(label, fontsize=13)
    ax.set_ylabel(r"Integrated Luminosity (pb$^{-1}$)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, linestyle="--", alpha=0.4)
    total = data["luminosity"].sum()
    ax.annotate(
        f"Total: {total:.2f} pb$^{{-1}}$",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
    )

axes[1].set_xlabel("Date (2021)", fontsize=11)
fig.suptitle("HPS 2021 Integrated Luminosity", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("luminosity_2021.pdf", bbox_inches="tight")
plt.savefig("luminosity_2021.png", dpi=150, bbox_inches="tight")
print(f"Main runs:   {main['luminosity'].sum():.3f} pb^-1  ({len(main)} runs)")
print(f"Moller runs: {moller['luminosity'].sum():.3f} pb^-1  ({len(moller)} runs)")
