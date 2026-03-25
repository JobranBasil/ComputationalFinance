# python -m src.plot_calibration

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
    INFORMED_RHO_VALUES,
    INFORMED_SIGMA_VALUES,
    MISPRICING_TARGET,
    run_informed_calibration_grid,
)

plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
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
# Noise trader calibration heatmap
# ---------------------------------------------------------------------------

def plot_calibration_grid(
    summary_df: pd.DataFrame,
    out_dir: str = ".",
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    summary_df = summary_df.sort_values(["psi", "rho_m"])

    spread_piv   = _pivot(summary_df, "mean_spread_mean")
    spread_valid = (spread_piv >= SPREAD_TARGET[0]) & (spread_piv <= SPREAD_TARGET[1])

    metrics = {
        "mean_spread_mean":    ("Mean Bid-Ask Spread",            "viridis_r", spread_valid),
        "volatility_mean":     ("Price Volatility (Log-Ret Std)", "viridis_r", None),
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

        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in piv.columns], fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in piv.index], fontsize=9)
        ax.set_xlabel(r"$\rho_m$ (market order probability)", fontsize=11)
        ax.set_ylabel(r"$\psi$ (sign persistence)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.values[i, j]
                fmt = f"{val:.5f}" if col in ("mean_spread_mean", "volatility_mean") else f"{val:.1f}"
                ax.text(j, i, fmt, ha="center", va="center",
                        color="white", fontsize=7.5, fontweight="bold")

        # hatch and border only on spread panel
        if col == "mean_spread_mean":
            _hatch_invalid(ax, piv, spread_valid)
            _border_valid(ax, piv, spread_valid)

        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

    valid_patch = mpatches.Patch(
        edgecolor="red", facecolor="none",
        linewidth=2.5, label="Within spread target"
    )
    hatch_patch = mpatches.Patch(
        facecolor="none", hatch="////",
        edgecolor="grey", label="Outside spread target"
    )
    fig.legend(
        handles=[valid_patch, hatch_patch],
        loc="lower center", ncol=2,
        fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        r"Calibration Grid Search: $\psi$ vs $\rho_m$"
        f"\nSpread target: [{SPREAD_TARGET[0]:.3f}, {SPREAD_TARGET[1]:.3f}]",
        fontsize=13, y=1.01, fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, "calibration_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {out_path}")


# ---------------------------------------------------------------------------
# Noise trader calibration lineplots
# ---------------------------------------------------------------------------

def plot_calibration_lineplots(
    summary_df: pd.DataFrame,
    out_dir: str = ".",
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.cm as cm

    psi_values = sorted(summary_df["psi"].unique())
    colours    = cm.tab10(np.linspace(0, 0.6, len(psi_values)))

    # spread only — volatility shown without target band
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = {
        "mean_spread_mean": "Mean Bid-Ask Spread",
        "volatility_mean":  "Price Volatility (Log-Ret Std)",
    }

    for ax, (col, title) in zip(axes, metrics.items()):
        for psi, colour in zip(psi_values, colours):
            sub    = summary_df[summary_df["psi"] == psi].sort_values("rho_m")
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

        # only shade spread panel
        if col == "mean_spread_mean":
            ax.axhspan(*SPREAD_TARGET, alpha=0.10, color="green",
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
# Informed trader calibration heatmap + lineplot
# ---------------------------------------------------------------------------

def plot_informed_calibration(
    summary_df: pd.DataFrame,
    out_dir: str = ".",
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.cm as cm

    summary_df = summary_df.sort_values(["rho", "sigma_s"])

    def _pivot_inf(col):
        return summary_df.pivot(index="rho", columns="sigma_s", values=col)

    valid_mask = (_pivot_inf("mispricing_valid")).astype(bool)

    metrics = {
        "avg_mispricing_mean": ("Avg Mispricing |V - M|",        "viridis_r"),
        "mean_spread_mean":    ("Mean Bid-Ask Spread",            "viridis_r"),
        "volatility_mean":     ("Price Volatility (Log-Ret Std)", "viridis_r"),
        "mean_depth_mean":     ("Mean Visible Depth",             "viridis"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, (col, (title, cmap)) in zip(axes.flat, metrics.items()):
        piv = _pivot_inf(col)

        im = ax.imshow(
            piv.values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.grid(False)

        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in piv.columns], fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in piv.index], fontsize=9)
        ax.set_xlabel(r"$\sigma_s$ (signal noise)", fontsize=11)
        ax.set_ylabel(r"$\rho$ (participation rate)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.values[i, j]
                fmt = f"{val:.4f}" if col in ("volatility_mean", "mean_spread_mean") else f"{val:.2f}"
                ax.text(j, i, fmt, ha="center", va="center",
                        color="white", fontsize=7.5, fontweight="bold")

        # red border + hatch
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                if valid_mask.iloc[i, j]:
                    ax.add_patch(mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="red",
                        linewidth=2.5, zorder=5,
                    ))

        if col == "avg_mispricing_mean":
            for i in range(len(piv.index)):
                for j in range(len(piv.columns)):
                    if not valid_mask.iloc[i, j]:
                        ax.add_patch(mpatches.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, hatch="////",
                            edgecolor="white", linewidth=0, alpha=0.6,
                        ))

        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

    valid_patch = mpatches.Patch(
        edgecolor="red", facecolor="none",
        linewidth=2.5, label="Within mispricing target"
    )
    hatch_patch = mpatches.Patch(
        facecolor="none", hatch="////",
        edgecolor="grey", label="Outside mispricing target"
    )
    fig.legend(
        handles=[valid_patch, hatch_patch],
        loc="lower center", ncol=2,
        fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        r"Informed Trader Calibration: $\rho$ vs $\sigma_s$"
        f"\nMispricing target: [{MISPRICING_TARGET[0]:.3f}, {MISPRICING_TARGET[1]:.3f}]",
        fontsize=13, y=1.01, fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(out_dir, "informed_calibration_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Informed calibration heatmap saved to {out_path}")

    # --- line plot ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    sigma_values = sorted(summary_df["sigma_s"].unique())
    colours      = cm.tab10(np.linspace(0, 0.6, len(sigma_values)))

    for sigma_s, colour in zip(sigma_values, colours):
        sub    = summary_df[summary_df["sigma_s"] == sigma_s].sort_values("rho")
        se_col = "avg_mispricing_se"
        ax.plot(
            sub["rho"], sub["avg_mispricing_mean"],
            marker="o", color=colour, linewidth=1.8,
            label=fr"$\sigma_s = {sigma_s:.2f}$",
        )
        if se_col in sub.columns:
            ax.fill_between(
                sub["rho"],
                sub["avg_mispricing_mean"] - sub[se_col],
                sub["avg_mispricing_mean"] + sub[se_col],
                alpha=0.15, color=colour,
            )

    ax.axhspan(*MISPRICING_TARGET, alpha=0.10, color="green",
               label="Calibration target")
    ax.set_xlabel(r"$\rho$ (participation rate)", fontsize=10)
    ax.set_ylabel(r"Avg Mispricing $|V - M|$", fontsize=10)
    ax.set_title(r"Avg Mispricing $|V - M|$", fontsize=11)
    ax.legend(title=r"Signal noise $\sigma_s$", fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        r"Informed Trader Calibration: $\rho$ vs $\sigma_s$",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "informed_calibration_lineplots.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Informed calibration line plots saved to {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    out_dir = os.path.join(os.path.dirname(__file__), "calibration_plots")
    os.makedirs(out_dir, exist_ok=True)

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

    informed_csv = os.path.join(out_dir, "informed_calibration_summary.csv")
    if os.path.exists(informed_csv):
        print(f"Loading existing informed calibration from {informed_csv}")
        informed_df = pd.read_csv(informed_csv)
    else:
        print("Running informed trader calibration grid...")
        informed_df = run_informed_calibration_grid(
            rho_values   = INFORMED_RHO_VALUES,
            sigma_values = INFORMED_SIGMA_VALUES,
            steps        = T_STEPS,
            n_seeds      = N_SEEDS,
            batch_size   = 10,
        )
        informed_df.to_csv(informed_csv, index=False)

    plot_informed_calibration(informed_df, out_dir=out_dir)