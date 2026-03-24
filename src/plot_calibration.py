"""
plot_calibration.py
===================
Heatmap plots for the calibration grid search results.

Run with:
    python -m src.plot_calibration
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from .calibration import (
    PSI_VALUES,
    RHO_M_VALUES,
    SPREAD_TARGET,
    VOLATILITY_TARGET,
    run_calibration_grid,
    N_SEEDS,
    T_STEPS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot summary df to (psi x rho_m) matrix for heatmap plotting."""
    return df.pivot(index="psi", columns="rho_m", values=value_col)


def _hatch_invalid(ax, piv, valid_mask, exclude_mask=None):
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            should_hatch = not valid_mask.iloc[i, j]
            if exclude_mask is not None:
                should_hatch = should_hatch and not exclude_mask.iloc[i, j]
            if should_hatch:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="////",
                    edgecolor="white", linewidth=0, alpha=0.6,
                ))


def _border_valid(ax, piv: pd.DataFrame, valid_mask: pd.DataFrame) -> None:
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            if valid_mask.iloc[i, j]:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.5,
                    zorder=5,
                ))


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_calibration_grid(
    summary_df: pd.DataFrame,
    out_dir: str = ".",
) -> None:
    """
    2x2 heatmap grid:
      - Mean Bid-Ask Spread
      - Price Volatility
      - Mean Visible Depth
      - Avg Mispricing

    Cells outside spread/volatility targets are hatched.
    Cells within BOTH targets get a red border on all panels.
    """

    os.makedirs(out_dir, exist_ok=True)

    # sort so pivot is consistent
    summary_df = summary_df.sort_values(["psi", "rho_m"])

    spread_piv = _pivot(summary_df, "mean_spread_mean")
    vol_piv    = _pivot(summary_df, "volatility_mean")

    spread_valid = (spread_piv >= SPREAD_TARGET[0]) & (spread_piv <= SPREAD_TARGET[1])
    vol_valid    = (vol_piv    >= VOLATILITY_TARGET[0]) & (vol_piv    <= VOLATILITY_TARGET[1])
    both_valid   = spread_valid & vol_valid

    metrics = {
        "mean_spread_mean":    ("Mean Bid-Ask Spread",            "viridis_r", spread_valid),
        "volatility_mean":     ("Price Volatility (Log-Ret Std)", "viridis_r", vol_valid),
        "mean_depth_mean":     ("Mean Visible Depth",             "viridis",   None),
        "avg_mispricing_mean": ("Avg Mispricing |V - M|",         "viridis_r", None),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, (col, (title, cmap, target_mask)) in zip(axes.flat, metrics.items()):
        piv = _pivot(summary_df, col)

        im = ax.imshow(
            piv.values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.grid(False)

        # axis labels
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in piv.columns], fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in piv.index], fontsize=9)
        ax.set_xlabel(r"$\rho_m$ (market order probability)", fontsize=11)
        ax.set_ylabel(r"$\psi$ (sign persistence)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        # annotate each cell
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.values[i, j]
                fmt = f"{val:.5f}" if col in ("mean_spread_mean", "volatility_mean") else f"{val:.1f}"
                ax.text(j, i, fmt, ha="center", va="center",
                        color="white", fontsize=7.5, fontweight="bold")

        # hatch out-of-target cells on spread and vol panels only
        if col == "mean_spread_mean":
            _hatch_invalid(ax, piv, spread_valid)
        elif col == "volatility_mean":
            _hatch_invalid(ax, piv, vol_valid, exclude_mask=both_valid)

        # red border = passes BOTH targets (on all panels)
        _border_valid(ax, piv, both_valid)

        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

    # legend
    valid_patch = mpatches.Patch(
        edgecolor="red", facecolor="none",
        linewidth=2.5, label="Within both targets (spread \& volatility)"
    )
    hatch_patch = mpatches.Patch(
        facecolor="none", hatch="////",
        edgecolor="grey", label="Outside panel-specific target"
    )
    fig.legend(
        handles=[valid_patch, hatch_patch],
        loc="lower center", ncol=2,
        fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    target_str = (
        f"Spread target: [{SPREAD_TARGET[0]:.3f}, {SPREAD_TARGET[1]:.3f}]  |  "
        f"Volatility target: [{VOLATILITY_TARGET[0]:.5f}, {VOLATILITY_TARGET[1]:.5f}]"
    )
    fig.suptitle(
        r"Calibration Grid Search: $\psi$ vs $\rho_m$" + f"\n{target_str}",
        fontsize=13, y=1.01, fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, "calibration_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {out_path}")


def plot_calibration_lineplots(
    summary_df: pd.DataFrame,
    out_dir: str = ".",
) -> None:
    """
    Line plots matching the style of plot_sensitivity() in analysis.py:
    x-axis = rho_m, lines coloured by psi, one panel per metric.
    """

    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.cm as cm

    psi_values = sorted(summary_df["psi"].unique())
    colours    = cm.tab10(np.linspace(0, 0.6, len(psi_values)))


    metrics = {
        "mean_spread_mean": "Mean Bid-Ask Spread",
        "volatility_mean":  "Price Volatility (Log-Ret Std)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (col, title) in zip(axes.flat, metrics.items()):
        for psi, colour in zip(psi_values, colours):
            sub = summary_df[summary_df["psi"] == psi].sort_values("rho_m")
            se_col = col.replace("_mean", "_se")
            ax.plot(
                sub["rho_m"], sub[col],
                marker="o", color=colour, linewidth=1.8,
                label=fr"$\psi = {psi:.1f}$",
            )
            if se_col in sub.columns:
                ax.fill_between(
                    sub["rho_m"],
                    sub[col] - sub[se_col],
                    sub[col] + sub[se_col],
                    alpha=0.15, color=colour,
                )

        # shade calibration target band on spread and vol panels
        if col == "mean_spread_mean":
            ax.axhspan(*SPREAD_TARGET, alpha=0.10, color="green",
                       label="Calibration target")
        elif col == "volatility_mean":
            ax.axhspan(*VOLATILITY_TARGET, alpha=0.10, color="green",
                       label="Calibration target")

        ax.set_xlabel(r"$\rho_m$ (market order probability)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(title=r"Sign persistence $\psi$", fontsize=7, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        r"Calibration: Market Quality vs $\rho_m$ for each $\psi$",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "calibration_lineplots.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Line plots saved to {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    out_dir = os.path.join(os.path.dirname(__file__), "calibration_plots")
    os.makedirs(out_dir, exist_ok=True)

    # run or load
    csv_path = os.path.join(out_dir, "calibration_summary.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}")
        summary_df = pd.read_csv(csv_path)
    else:
        print("No existing results found — running grid search now...")
        summary_df = run_calibration_grid(
            psi_values=PSI_VALUES,
            rho_m_values=RHO_M_VALUES,
            steps=T_STEPS,
            n_seeds=N_SEEDS,
            batch_size=10,
        )
        summary_df.to_csv(csv_path, index=False)

    plot_calibration_grid(summary_df, out_dir=out_dir)
    plot_calibration_lineplots(summary_df, out_dir=out_dir)

    print("\nDone. Outputs in:", out_dir)