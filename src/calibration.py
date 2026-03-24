"""
calibration.py
==============
Grid search over noise trader parameters (psi, rho_m) to identify
configurations that produce realistic baseline market quality.

Run with:
    python -m src.calibration
"""

from __future__ import annotations

import os
import logging
import itertools
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
)

import importlib
_dp_mod = importlib.import_module("src.dark_pool")
DarkPool = _dp_mod.DarkPool


# ---------------------------------------------------------------------------
# Calibration targets
# Based on baseline Table 2 in the paper and empirical microstructure norms:
# spread should be a few ticks (tick=0.01), volatility should be low
# ---------------------------------------------------------------------------

SPREAD_TARGET     = (0.03, 0.07)    # a few ticks either side of baseline 0.046
VOLATILITY_TARGET = (0.00030, 0.00075)   # log-return std, matches baseline 0.04%

PSI_VALUES   = [0.5, 0.6, 0.7, 0.8, 0.9]    # sign persistence
RHO_M_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]    # market order probability

N_SEEDS  = 10
T_STEPS  = 1000


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_one_calibration(
    psi:    float,
    rho_m:  float,
    steps:  int,
    seed:   int,
) -> dict:
    """
    Single calibration run with noise trader parameters varied.
    All other parameters fixed at their baseline values.
    dark_fraction=0, iceberg_prob=0 to match the lambda=0, mu=0 baseline.
    """

    fundamental = FundamentalProcess(
        start=100.075,
        sigma_v=0.03,
        rng=np.random.default_rng(seed),
    )

    book = OrderBook(tick=0.01, max_depth_levels=20)
    seed_initial_book(
        book,
        best_bid=100.05,
        best_ask=100.10,
        levels=10,
        rng=np.random.default_rng(seed + 1),
    )

    agents = [
        NoiseTrader(trader_id=1,  rng=np.random.default_rng(seed + 10),
                    sign_persistence=psi, market_prob=rho_m),
        NoiseTrader(trader_id=2,  rng=np.random.default_rng(seed + 11),
                    sign_persistence=psi, market_prob=rho_m),
        NoiseTrader(trader_id=3,  rng=np.random.default_rng(seed + 12),
                    sign_persistence=psi, market_prob=rho_m),
        NoiseTrader(trader_id=4,  rng=np.random.default_rng(seed + 13),
                    sign_persistence=psi, market_prob=rho_m),
        NoiseTrader(trader_id=5,  rng=np.random.default_rng(seed + 14),
                    sign_persistence=psi, market_prob=rho_m),
        NoiseTrader(trader_id=6,  rng=np.random.default_rng(seed + 15),
                    sign_persistence=psi, market_prob=rho_m),
        MarketMaker(trader_id=7,  rng=np.random.default_rng(seed + 16)),
        InstitutionalTrader(
            trader_id=8,
            rng=np.random.default_rng(seed + 17),
            use_iceberg_prob=0.0,
            dark_fraction=0.0,
        ),
        InstitutionalTrader(
            trader_id=9,
            rng=np.random.default_rng(seed + 18),
            use_iceberg_prob=0.0,
            dark_fraction=0.0,
        ),
        InformedTrader(
            trader_id=10,
            rng=np.random.default_rng(seed + 19),
            fundamental=fundamental,
            sigma_s=0.08,
            participation_rate=0.4,
            dark_fraction=0.0,
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
        "psi":            psi,
        "rho_m":          rho_m,
        "seed":           seed,
        "mean_spread":    float(book_df["Spread"].mean()),
        "mean_depth":     float(
            (book_df["TotalBidDepth"] + book_df["TotalAskDepth"]).mean()
        ),
        "volatility":     float(np.std(log_rets)) if len(log_rets) > 0 else np.nan,
        "avg_mispricing": avg_mispricing,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_calibration_grid(
    psi_values:   list | None = None,
    rho_m_values: list | None = None,
    steps:        int  = T_STEPS,
    n_seeds:      int  = N_SEEDS,
    base_seed:    int  = 0,
    batch_size:   int  = 10,
    silent:       bool = False,
) -> pd.DataFrame:
    """
    Runs the calibration grid search over (psi, rho_m) pairs.
    Returns a summary DataFrame with mean and SE per configuration.
    """

    if psi_values is None:
        psi_values = PSI_VALUES
    if rho_m_values is None:
        rho_m_values = RHO_M_VALUES

    all_args = [
        (psi, rho_m, steps, base_seed + i * 100 + run)
        for i, (psi, rho_m) in enumerate(itertools.product(psi_values, rho_m_values))
        for run in range(n_seeds)
    ]

    all_results = []
    for batch_start in range(0, len(all_args), batch_size):
        batch = all_args[batch_start : batch_start + batch_size]
        if not silent:
            print(
                f"Calibration batch [{batch_start + 1}--"
                f"{batch_start + len(batch)}/{len(all_args)}]",
                flush=True,
            )
        with Pool(processes=min(cpu_count(), len(batch))) as pool:
            batch_results = pool.starmap(_run_one_calibration, batch)
        all_results.extend(batch_results)

    metric_cols = ["mean_spread", "mean_depth", "volatility", "avg_mispricing"]
    raw_df      = pd.DataFrame(all_results)
    grouped     = raw_df.groupby(["psi", "rho_m"])

    summary_mean = grouped[metric_cols].mean().add_suffix("_mean")
    summary_se   = grouped[metric_cols].sem().add_suffix("_se")
    summary      = pd.concat([summary_mean, summary_se], axis=1).reset_index()
    summary["n_seeds"] = n_seeds

    # flag cells within calibration targets
    summary["spread_valid"]     = summary["mean_spread_mean"].between(*SPREAD_TARGET)
    summary["volatility_valid"] = summary["volatility_mean"].between(*VOLATILITY_TARGET)
    summary["valid"]            = summary["spread_valid"] & summary["volatility_valid"]

    return summary


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    out_dir = os.path.join(os.path.dirname(__file__), "calibration_plots")
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== CALIBRATION GRID SEARCH ===")
    print(f"Grid: psi={PSI_VALUES}, rho_m={RHO_M_VALUES}")
    print(f"Seeds per config: {N_SEEDS}, Steps per run: {T_STEPS}")
    print(f"Total runs: {len(PSI_VALUES) * len(RHO_M_VALUES) * N_SEEDS}\n")

    summary_df = run_calibration_grid(
        psi_values   = PSI_VALUES,
        rho_m_values = RHO_M_VALUES,
        steps        = T_STEPS,
        n_seeds      = N_SEEDS,
        batch_size   = 10,
    )

    summary_df.to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)
    print("\nValid configurations (within spread AND volatility targets):")
    valid = summary_df[summary_df["valid"]]
    if len(valid) > 0:
        print(valid[["psi", "rho_m",
                      "mean_spread_mean", "volatility_mean",
                      "mean_depth_mean",  "avg_mispricing_mean"]].to_string(index=False))
    else:
        print("  None found — consider widening SPREAD_TARGET / VOLATILITY_TARGET.")

    print(f"\nFull results saved to {out_dir}/calibration_summary.csv")