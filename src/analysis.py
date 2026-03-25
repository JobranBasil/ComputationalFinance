# python -m src.analysis

from __future__ import annotations

import os
import logging
import itertools
from typing import List
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .fundemental import FundamentalProcess
from .orderbook import OrderBook, Order
from .agents import (
    NoiseTrader,
    MarketMaker,
    InstitutionalTrader,
    MarketMakerAS,
    InformedTrader,
)
from .run import (
    run_simulation,
    seed_initial_book,
    add_market_analytics,
    apply_action,
)

import importlib
_dp_mod = importlib.import_module("src.dark_pool")
DarkPool = _dp_mod.DarkPool

plt.style.use("seaborn-v0_8-whitegrid")


# ---------------------------------------------------------------------------
# Single seeded run
# ---------------------------------------------------------------------------

def _build_and_run(seed: int, steps: int = 1000) -> dict:
    # single seeded run

    fundamental = FundamentalProcess(
        start=100.075,
        sigma_v=0.03,
        rng=np.random.default_rng(seed),
    )

    book = OrderBook(tick=0.01, max_depth_levels=20)
    seed_initial_book(book, best_bid=100.05, best_ask=100.10, levels=10,
                      rng=np.random.default_rng(seed + 1))

    agents = [
        NoiseTrader(trader_id=1,  rng=np.random.default_rng(seed + 10)),
        NoiseTrader(trader_id=2,  rng=np.random.default_rng(seed + 11)),
        NoiseTrader(trader_id=3,  rng=np.random.default_rng(seed + 12)),
        NoiseTrader(trader_id=4,  rng=np.random.default_rng(seed + 13)),
        NoiseTrader(trader_id=5,  rng=np.random.default_rng(seed + 14)),
        NoiseTrader(trader_id=6,  rng=np.random.default_rng(seed + 15)),
        MarketMaker(trader_id=7,  rng=np.random.default_rng(seed + 16)),
        InstitutionalTrader(trader_id=8,  use_iceberg_prob=0, dark_fraction=0,
                            rng=np.random.default_rng(seed + 17)),
        InstitutionalTrader(trader_id=9,  use_iceberg_prob=0, dark_fraction=0,
                            rng=np.random.default_rng(seed + 18)),
        InformedTrader(
            trader_id=10,
            rng=np.random.default_rng(seed + 19),
            fundamental=fundamental,
            sigma_s=0.08,
            participation_rate=0.4,
            dark_fraction=0,
        ),
        MarketMakerAS(
            trader_id=11,
            rng=np.random.default_rng(seed + 20),
            horizon=5000,
            kappa=50,
            gamma=0.1,
            sigma=0.05,
        ),
    ]

    dark_pool = DarkPool(
        lit_orderbook=book,
        max_resting_ticks=50,
        routing_delay=5,
        tape_delay=5,
    )

    _, book_df, trades_df, dp_trades_df, _ = run_simulation(
        book, agents, dark_pool, steps=steps, fundamental=fundamental,
    )

    book_df  = add_market_analytics(book_df, vol_window=10)
    mids     = book_df["Mid"].dropna().values
    log_rets = np.diff(np.log(mids[mids > 0])) if len(mids) > 1 else np.array([])

    fund_series    = np.array(fundamental.history)[:len(book_df)]
    mid_series     = book_df["Mid"].values
    avg_mispricing = float(np.nanmean(np.abs(fund_series - mid_series)))

    return {
        "seed":            seed,
        "mean_spread":     float(book_df["Spread"].mean()),
        "std_spread":      float(book_df["Spread"].std()),
        "mean_depth":      float(
            (book_df["TotalBidDepth"] + book_df["TotalAskDepth"]).mean()
        ),
        "volatility":      float(np.std(log_rets)) if len(log_rets) > 0 else np.nan,
        "avg_mispricing":  avg_mispricing,
        "total_lit_vol":   int(book_df["TradeVolume"].sum()),
        "total_dp_vol":    int(dp_trades_df["Qty"].sum()) if len(dp_trades_df) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Multiple baseline runs
# ---------------------------------------------------------------------------

def run_baseline(
    n_runs:     int  = 20,
    steps:      int  = 1000,
    base_seed:  int  = 0,
    batch_size: int  = 5,
    silent:     bool = False,
) -> pd.DataFrame:

    seeds = [base_seed + i for i in range(n_runs)]
    raw   = []

    for batch_start in range(0, n_runs, batch_size):
        batch_seeds = seeds[batch_start : batch_start + batch_size]
        if not silent:
            print(
                f"Baseline batch [{batch_start+1}--{batch_start+len(batch_seeds)}/{n_runs}]",
                flush=True,
            )
        with Pool(processes=min(cpu_count(), len(batch_seeds))) as pool:
            batch_results = pool.starmap(_build_and_run, [(s, steps) for s in batch_seeds])
        raw.extend(batch_results)

    return pd.DataFrame(raw)


def plot_baseline(results_df: pd.DataFrame, out_dir: str = ".") -> None:
    # 2x2 boxplots

    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "mean_spread":    "Mean Bid-Ask Spread",
        "mean_depth":     "Mean Total Visible Depth",
        "volatility":     "Price Volatility (Log-Return Std)",
        "avg_mispricing": "Avg |Fundamental $-$ Mid| (Mispricing)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, (col, title) in zip(axes.flat, metrics.items()):
        data = results_df[col].dropna()
        mean = data.mean()
        se   = data.std() / np.sqrt(len(data))

        ax.boxplot(data, widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.5))
        ax.scatter([1] * len(data), data, color="steelblue", alpha=0.6, zorder=3)
        ax.axhline(mean, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean:.5f}\nSE = {se:.5f}")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Baseline Market Quality Distribution ({len(results_df)} runs)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "baseline_distribution.png"), dpi=150)
    plt.close(fig)

    # summary csv with SE included
    scalar_cols = list(metrics.keys())
    summary = results_df[scalar_cols].agg(["mean", "std", "min", "max"])
    summary.loc["se"] = results_df[scalar_cols].std() / np.sqrt(len(results_df))
    summary.to_csv(os.path.join(out_dir, "baseline_summary.csv"))
    print(f"Baseline plots and summary saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Validation: informed trader participation rate vs mispricing
# ---------------------------------------------------------------------------

def _build_and_run_informed_rate(seed: int, participation_rate: float,
                                  steps: int = 1000) -> dict:
    # same model, configurable informed rate

    fundamental = FundamentalProcess(
        start=100.075,
        sigma_v=0.03,
        rng=np.random.default_rng(seed),
    )

    book = OrderBook(tick=0.01, max_depth_levels=20)
    seed_initial_book(book, best_bid=100.05, best_ask=100.10, levels=10,
                      rng=np.random.default_rng(seed + 1))

    agents = [
        NoiseTrader(trader_id=1,  rng=np.random.default_rng(seed + 10)),
        NoiseTrader(trader_id=2,  rng=np.random.default_rng(seed + 11)),
        NoiseTrader(trader_id=3,  rng=np.random.default_rng(seed + 12)),
        NoiseTrader(trader_id=4,  rng=np.random.default_rng(seed + 13)),
        NoiseTrader(trader_id=5,  rng=np.random.default_rng(seed + 14)),
        NoiseTrader(trader_id=6,  rng=np.random.default_rng(seed + 15)),
        MarketMaker(trader_id=7,  rng=np.random.default_rng(seed + 16)),
        InstitutionalTrader(trader_id=8,  use_iceberg_prob=0, dark_fraction=0,
                            rng=np.random.default_rng(seed + 17)),
        InstitutionalTrader(trader_id=9,  use_iceberg_prob=0, dark_fraction=0,
                            rng=np.random.default_rng(seed + 18)),
        InformedTrader(
            trader_id=10,
            rng=np.random.default_rng(seed + 19),
            fundamental=fundamental,
            sigma_s=0.08,
            participation_rate=participation_rate,
            dark_fraction=0,
        ),
        MarketMakerAS(
            trader_id=11,
            rng=np.random.default_rng(seed + 20),
            horizon=5000,
            kappa=50,
            gamma=0.1,
            sigma=0.05,
        ),
    ]

    dark_pool = DarkPool(
        lit_orderbook=book,
        max_resting_ticks=50,
        routing_delay=5,
        tape_delay=5,
    )

    _, book_df, _, _, _ = run_simulation(
        book, agents, dark_pool, steps=steps, fundamental=fundamental,
    )

    book_df = add_market_analytics(book_df, vol_window=10)

    fund_series    = np.array(fundamental.history)[:len(book_df)]
    mid_series     = book_df["Mid"].values
    avg_mispricing = float(np.nanmean(np.abs(fund_series - mid_series)))

    return {
        "seed":               seed,
        "participation_rate": participation_rate,
        "avg_mispricing":     avg_mispricing,
    }


def run_validation_informed_rate(
    participation_rates: List[float] | None = None,
    n_runs:              int  = 20,
    steps:               int  = 1000,
    base_seed:           int  = 200,
    batch_size:          int  = 5,
    silent:              bool = False,
) -> pd.DataFrame:
    # MC validation: informed rate vs mispricing

    if participation_rates is None:
        participation_rates = [0.4, 1.0]

    all_args = [
        (base_seed + i + r_idx * 1000, rate, steps)
        for r_idx, rate in enumerate(participation_rates)
        for i in range(n_runs)
    ]

    all_results = []
    for batch_start in range(0, len(all_args), batch_size):
        batch = all_args[batch_start : batch_start + batch_size]
        if not silent:
            print(
                f"Validation batch [{batch_start+1}--{batch_start+len(batch)}/{len(all_args)}]",
                flush=True,
            )
        with Pool(processes=min(cpu_count(), len(batch))) as pool:
            batch_results = pool.starmap(_build_and_run_informed_rate, batch)
        all_results.extend(batch_results)

    return pd.DataFrame(all_results)


def plot_validation_informed_rate(
    results_df: pd.DataFrame,
    out_dir:    str = ".",
) -> None:
    # boxplots per participation rate

    os.makedirs(out_dir, exist_ok=True)

    rates  = sorted(results_df["participation_rate"].unique())
    data   = [results_df[results_df["participation_rate"] == r]["avg_mispricing"].values
               for r in rates]
    means  = [d.mean() for d in data]
    labels = [f"$\\rho = {r:.1f}$" for r in rates]

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.4,
        boxprops=dict(alpha=0.6),
    )

    colours = ["steelblue", "crimson"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)

    for i, (mean, colour) in enumerate(zip(means, colours), start=1):
        ax.axhline(mean, xmin=(i - 1) / len(rates) + 0.05,
                   xmax=i / len(rates) - 0.05,
                   color=colour, linestyle="--", linewidth=1.5,
                   label=f"Mean ($\\rho={rates[i-1]:.1f}$) = {mean:.4f}")
        ax.scatter([i] * len(data[i - 1]), data[i - 1],
                   color=colour, alpha=0.5, zorder=3)

    ax.set_ylabel("Avg $|$Fundamental $-$ Mid$|$ (Mispricing)", fontsize=12)
    ax.set_xlabel("Informed Trader Participation Rate", fontsize=12)
    ax.set_title(
        "Effect of Informed Trader Participation Rate on Mispricing\n"
        f"({len(results_df) // len(rates)} Monte Carlo runs per rate)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation_informed_rate.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: validation_informed_rate.png")

    # print summary
    for rate, d, mean in zip(rates, data, means):
        se = d.std() / np.sqrt(len(d))
        print(f"  rho={rate:.1f}  mean_mispricing={mean:.4f}  SE={se:.4f}")


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def _run_one_sensitivity(
    dark_frac:    float,
    iceberg_prob: float,
    steps:        int,
    seed:         int,
) -> dict:
    # single sensitivity run

    fundamental = FundamentalProcess(
        start=100.075,
        sigma_v=0.03,
        rng=np.random.default_rng(seed),
    )

    book = OrderBook(tick=0.01, max_depth_levels=20)
    seed_initial_book(book, best_bid=100.05, best_ask=100.10, levels=10,
                      rng=np.random.default_rng(seed + 1))

    agents = [
        NoiseTrader(trader_id=1,  rng=np.random.default_rng(seed + 10)),
        NoiseTrader(trader_id=2,  rng=np.random.default_rng(seed + 11)),
        NoiseTrader(trader_id=3,  rng=np.random.default_rng(seed + 12)),
        NoiseTrader(trader_id=4,  rng=np.random.default_rng(seed + 13)),
        NoiseTrader(trader_id=5,  rng=np.random.default_rng(seed + 14)),
        NoiseTrader(trader_id=6,  rng=np.random.default_rng(seed + 15)),
        MarketMaker(trader_id=7,  rng=np.random.default_rng(seed + 16)),
        InstitutionalTrader(
            trader_id=8,
            rng=np.random.default_rng(seed + 17),
            use_iceberg_prob=iceberg_prob,
            dark_fraction=dark_frac,
        ),
        InstitutionalTrader(
            trader_id=9,
            rng=np.random.default_rng(seed + 18),
            use_iceberg_prob=iceberg_prob,
            dark_fraction=dark_frac,
        ),
        InformedTrader(
            trader_id=10,
            rng=np.random.default_rng(seed + 19),
            fundamental=fundamental,
            sigma_s=0.08,
            participation_rate=0.4,
            dark_fraction=dark_frac,
        ),
        MarketMakerAS(
            trader_id=11,
            rng=np.random.default_rng(seed + 20),
            horizon=5000,
            kappa=50,
            gamma=0.1,
            sigma=0.05,
        ),
    ]

    dark_pool = DarkPool(
        lit_orderbook=book,
        max_resting_ticks=50,
        routing_delay=5,
        tape_delay=5,
    )

    _, book_df, _, _, _ = run_simulation(
        book, agents, dark_pool, steps=steps, fundamental=fundamental,
    )

    book_df  = add_market_analytics(book_df, vol_window=10)
    mids     = book_df["Mid"].dropna().values
    log_rets = np.diff(np.log(mids[mids > 0])) if len(mids) > 1 else np.array([])

    fund_series    = np.array(fundamental.history)[:len(book_df)]
    mid_series     = book_df["Mid"].values
    avg_mispricing = float(np.nanmean(np.abs(fund_series - mid_series)))

    return {
        "dark_frac":      dark_frac,
        "iceberg_prob":   iceberg_prob,
        "mean_spread":    float(book_df["Spread"].mean()),
        "mean_depth":     float(
            (book_df["TotalBidDepth"] + book_df["TotalAskDepth"]).mean()
        ),
        "volatility":     float(np.std(log_rets)) if len(log_rets) > 0 else np.nan,
        "avg_mispricing": avg_mispricing,
    }


def run_sensitivity(
    dark_frac_values:    List[float] | None = None,
    iceberg_prob_values: List[float] | None = None,
    steps:               int  = 1000,
    n_runs:              int  = 5,
    base_seed:           int  = 42,
    batch_size:          int  = 10,
    silent:              bool = False,
) -> pd.DataFrame:
    # grid search: dark_frac x iceberg_prob

    if dark_frac_values is None:
        dark_frac_values = np.arange(0, 1.01, 0.2).tolist()
    if iceberg_prob_values is None:
        iceberg_prob_values = np.arange(0, 1.01, 0.2).tolist()

    all_args = [
        (df, ip, steps, base_seed + i * 100 + run)
        for i, (df, ip) in enumerate(
            itertools.product(dark_frac_values, iceberg_prob_values)
        )
        for run in range(n_runs)
    ]

    all_results = []
    for batch_start in range(0, len(all_args), batch_size):
        batch = all_args[batch_start : batch_start + batch_size]
        if not silent:
            print(
                f"Sensitivity batch [{batch_start+1}--{batch_start+len(batch)}/{len(all_args)}]",
                flush=True,
            )
        with Pool(processes=min(cpu_count(), len(batch))) as pool:
            batch_results = pool.starmap(_run_one_sensitivity, batch)
        all_results.extend(batch_results)

    metric_cols = ["mean_spread", "mean_depth", "volatility", "avg_mispricing"]
    results_df  = pd.DataFrame(all_results)
    grouped     = results_df.groupby(["dark_frac", "iceberg_prob"])

    # mean and SE
    summary_mean = grouped[metric_cols].mean().add_suffix("_mean")
    summary_se   = grouped[metric_cols].sem().add_suffix("_se")
    summary      = pd.concat([summary_mean, summary_se], axis=1).reset_index()
    summary["n_runs"] = n_runs

    return summary


def plot_sensitivity(results_df: pd.DataFrame, out_dir: str = ".") -> None:
    # line plot per metric

    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "mean_spread":    "Mean Bid-Ask Spread",
        "mean_depth":     "Mean Total Visible Depth (Liquidity)",
        "volatility":     "Price Volatility (Log-Return Std)",
        "avg_mispricing": "Avg $|$Fundamental $-$ Mid$|$ (Mispricing)",
    }

    iceberg_probs = sorted(results_df["iceberg_prob"].unique())
    colours = cm.tab10(np.linspace(0, 0.6, len(iceberg_probs)))

    def _draw_metric(ax, col, title):
        mean_col = f"{col}_mean"
        for ip, colour in zip(iceberg_probs, colours):
            sub  = results_df[results_df["iceberg_prob"] == ip].sort_values("dark_frac")
            ax.plot(
                sub["dark_frac"],
                sub[mean_col],
                marker="o",
                color=colour,
                linewidth=1.8,
                label=f"$\\mu = {ip:.2f}$",
            )
        ax.set_xlabel("Dark Pool Routing Fraction ($\\lambda$)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(title="Iceberg Probability", fontsize=7, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)

    # individual plots
    for col, title in metrics.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_metric(ax, col, title)
        ax.set_title(f"{title} vs Dark Pool Routing Fraction ($\\lambda$)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"sensitivity_{col}.png"), dpi=150)
        plt.close(fig)

    # 2x2 summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flat, metrics.items()):
        _draw_metric(ax, col, title)

    fig.suptitle(
        "Market Quality Sensitivity Analysis",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_summary.png"), dpi=150)
    plt.close(fig)

    # save results with SE
    results_df.to_csv(os.path.join(out_dir, "sensitivity_results.csv"), index=False)
    print(f"Sensitivity plots and results (with SE) saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    out_dir = os.path.join(os.path.dirname(__file__), "analysis_plots")
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Baseline: 20 independent runs ---
    print("\n=== BASELINE (20 runs) ===")
    baseline_df = run_baseline(n_runs=20, steps=1000, base_seed=0, batch_size=5)
    baseline_df.to_csv(os.path.join(out_dir, "baseline_runs.csv"), index=False)
    print(baseline_df[["mean_spread", "mean_depth", "volatility", "avg_mispricing"]].describe())
    plot_baseline(baseline_df, out_dir=out_dir)

    # --- 2. Validation: informed trader participation rate ---
    print("\n=== VALIDATION: Informed Trader Participation Rate ===")
    validation_df = run_validation_informed_rate(
        participation_rates = [0.4, 1.0],
        n_runs              = 20,
        steps               = 1000,
        base_seed           = 200,
        batch_size          = 5,
    )
    validation_df.to_csv(os.path.join(out_dir, "validation_informed_rate.csv"), index=False)
    plot_validation_informed_rate(validation_df, out_dir=out_dir)

    # --- 3. Sensitivity analysis ---
    print("\n=== SENSITIVITY ANALYSIS ===")
    sensitivity_df = run_sensitivity(
        dark_frac_values    = np.arange(0, 1.01, 0.2).tolist(),
        iceberg_prob_values = np.arange(0, 1.01, 0.2).tolist(),
        steps               = 1000,
        n_runs              = 10,
        batch_size          = 10,
    )
    plot_sensitivity(sensitivity_df, out_dir=out_dir)

    print("\nAll done. Outputs in:", out_dir)