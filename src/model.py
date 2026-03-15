"""
Mesa Model for the order-book ABM with dark pool and fundamental process.

OrderBookModel owns: OrderBook, DarkPool, FundamentalProcess, all agents, DataCollector.
"""

from __future__ import annotations

import os
import importlib
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mesa

from .orderbook import OrderBook, Order
from .fundemental import FundamentalProcess
from .agents import (
    BaseAgent, NoiseTrader, MarketMaker,
    InstitutionalTrader, MarketMakerAS, InformedTrader,
    Action,
)

_dp_mod = importlib.import_module("src.dark_pool")
DarkPool = _dp_mod.DarkPool


# ---------------------------------------------------------------------------
# Order book helpers
# ---------------------------------------------------------------------------

def apply_action(book: OrderBook, action) -> list:
    """Apply an agent action to the order book. Returns list of trades."""
    if action is None:
        return []
    if isinstance(action, list):
        all_trades = []
        for sub_action in action:
            all_trades.extend(apply_action(book, sub_action))
        return all_trades
    if isinstance(action, tuple) and action[0] == "cancel":
        book.cancel(action[1])
        return []
    if isinstance(action, Order):
        return book.execute_market(action) if action.price is None else book.add_limit(action)
    raise TypeError(f"Unknown action type: {type(action)}")


def seed_initial_book(book, best_bid=100.0, best_ask=101.0, levels=5, rng=None):
    rng = rng or np.random.default_rng(42)
    for i in range(levels):
        book.add_limit_post_only(Order(
            order_id=10_000 + i, trader_id=-1, side="buy",
            qty=int(rng.integers(1, 10)), price=best_bid - book.tick * i, ts=0,
        ))
        book.add_limit_post_only(Order(
            order_id=20_000 + i, trader_id=-1, side="sell",
            qty=int(rng.integers(1, 10)), price=best_ask + book.tick * i, ts=0,
        ))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def best_level_depth(book):
    bb, ba = book.best_bid(), book.best_ask()
    bid_qty = sum(o.qty for o in book.bids.get(bb, [])) if np.isfinite(bb) else 0
    ask_qty = sum(o.qty for o in book.asks.get(ba, [])) if np.isfinite(ba) else 0
    return bid_qty, ask_qty


def total_visible_depth(book):
    bid_total = sum(o.qty for p in book.bid_prices for o in book.bids[p])
    ask_total = sum(o.qty for p in book.ask_prices for o in book.asks[p])
    return bid_total, ask_total


def top_n_depth(book, n=5):
    bid_depth = sum(q for _, q in book.top_n_levels("buy", n))
    ask_depth = sum(q for _, q in book.top_n_levels("sell", n))
    return bid_depth, ask_depth


def top_n_obi(book, n=5):
    bid_depth, ask_depth = top_n_depth(book, n)
    denom = bid_depth + ask_depth
    return (bid_depth - ask_depth) / denom if denom > 0 else 0.0


def microprice(book):
    bb, ba = book.best_bid(), book.best_ask()
    if not np.isfinite(bb) or not np.isfinite(ba):
        return np.nan
    qb = sum(o.qty for o in book.bids.get(bb, []))
    qa = sum(o.qty for o in book.asks.get(ba, []))
    denom = qb + qa
    return (ba * qb + bb * qa) / denom if denom > 0 else np.nan


def trade_volume(trades):
    return sum(tr.qty for tr in trades)


def signed_trade_volume(trades):
    return sum(tr.qty if tr.aggressor_side == "buy" else -tr.qty for tr in trades)


def vwap(trades):
    if not trades:
        return np.nan
    total_qty = sum(tr.qty for tr in trades)
    return sum(tr.price * tr.qty for tr in trades) / total_qty if total_qty > 0 else np.nan


def add_market_analytics(df, vol_window=10):
    df = df.copy()
    df["MidReturn"] = df["Mid"].diff()
    df["LogMidReturn"] = np.log(df["Mid"] / df["Mid"].shift(1))
    df["RollingVol"] = df["LogMidReturn"].rolling(vol_window).std()
    df["SpreadChange"] = df["Spread"].diff()
    df["MicropriceReturn"] = df["Microprice"].diff()
    return df


# ---------------------------------------------------------------------------
# Plotting (moved from original run.py, unchanged)
# ---------------------------------------------------------------------------

def plot_series(series, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series, label=title)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_snapshots(snapshots, out_path):
    fig, axes = plt.subplots(len(snapshots), 1, figsize=(8, 3 * len(snapshots)))
    if len(snapshots) == 1:
        axes = [axes]
    tick = 0.01
    if snapshots and len(snapshots[0].get("bids", [])) >= 2:
        tick = abs(snapshots[0]["bids"][0][0] - snapshots[0]["bids"][1][0]) or 0.01
    bar_width = tick * 0.4
    half_width = tick * 22
    for ax, snap in zip(axes, snapshots):
        bids, asks = snap["bids"], snap["asks"]
        if bids:
            bp, bq = zip(*bids)
            ax.bar(bp, bq, width=bar_width, label="Bids")
        else:
            bp, bq = [], []
        if asks:
            ap, aq = zip(*asks)
            ax.bar(ap, [-q for q in aq], width=bar_width, label="Asks")
        else:
            ap, aq = [], []
        ax.axhline(0, linewidth=1)
        if bids and asks:
            mid = (ap[0] + bp[0]) / 2
            ax.axvline(mid, linewidth=1, color="black", linestyle="--")
            ax.set_xlim(mid - half_width, mid + half_width)
        ax.set_title(f"t={snap['t']}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_demand_curve(bids, asks, out_path):
    bid_prices = [p for p, _ in bids]
    ask_prices = [p for p, _ in asks]
    bid_cum = [sum(q for p, q in bids if p >= price) for price in bid_prices]
    ask_cum = [sum(q for p, q in asks if p <= price) for price in ask_prices]
    fig, ax = plt.subplots()
    ax.step(bid_cum, bid_prices, where="post", color="blue", label="Bids")
    ax.step(ask_cum, ask_prices, where="post", color="red", label="Asks")
    ax.set_xlabel("Cumulative Quantity")
    ax.set_ylabel("Price")
    ax.set_title("Demand Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mmAS_inventory(book_df, out_path):
    if "MMAS_Inventory" not in book_df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(book_df["t"], book_df["MMAS_Inventory"], label="MMAS Inventory", color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(book_df["t"], book_df["MMAS_Inventory"], 0,
                    where=book_df["MMAS_Inventory"] > 0, alpha=0.2, color="green", label="Long")
    ax.fill_between(book_df["t"], book_df["MMAS_Inventory"], 0,
                    where=book_df["MMAS_Inventory"] < 0, alpha=0.2, color="red", label="Short")
    ax.set_title("MarketMakerAS Inventory Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Inventory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mid_vs_fundamental(book_df, fundamental_history, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    fund_series = fundamental_history[:len(book_df)]
    ax.plot(book_df["t"], book_df["Mid"], label="Mid Price", color="steelblue", linewidth=1.2)
    ax.plot(book_df["t"], fund_series, label="Fundamental Value", color="crimson",
            linewidth=1.2, linestyle="--")
    ax.fill_between(book_df["t"], book_df["Mid"], fund_series, alpha=0.15, color="purple", label="Mispricing")
    ax.set_title("Mid Price vs Fundamental Value")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mesa Model
# ---------------------------------------------------------------------------

class OrderBookModel(mesa.Model):
    def __init__(
        self,
        steps_to_run: int = 1000,
        seed: int = 42,
        tick: float = 0.01,
        max_depth_levels: int = 20,
        best_bid: float = 100.05,
        best_ask: float = 100.10,
        init_levels: int = 10,
    ):
        super().__init__(rng=seed)

        self.steps_to_run = steps_to_run
        self.current_step = 0
        self._sim_rng = np.random.default_rng(42)

        # ---- Fundamental process ----
        self.fundamental = FundamentalProcess(
            start=100.075, sigma_v=0.03, rng=np.random.default_rng(99),
        )

        # ---- Order book ----
        self.book = OrderBook(tick=tick, max_depth_levels=max_depth_levels)
        seed_initial_book(self.book, best_bid=best_bid, best_ask=best_ask,
                          levels=init_levels, rng=np.random.default_rng(seed))

        # ---- Dark pool ----
        self.dark_pool = DarkPool(
            lit_orderbook=self.book,
            max_resting_ticks=50, routing_delay=5, tape_delay=5,
        )

        # ---- Agents (same as original run.py main()) ----
        NoiseTrader(model=self, trader_id=1, rng=np.random.default_rng(1))
        NoiseTrader(model=self, trader_id=5, rng=np.random.default_rng(5))
        NoiseTrader(model=self, trader_id=6, rng=np.random.default_rng(6))
        MarketMaker(model=self, trader_id=2, rng=np.random.default_rng(2))
        InstitutionalTrader(model=self, trader_id=3, rng=np.random.default_rng(3))
        InstitutionalTrader(model=self, trader_id=8, rng=np.random.default_rng(8))
        InformedTrader(
            model=self, trader_id=7, rng=np.random.default_rng(7),
            fundamental=self.fundamental,
            sigma_s=0.08, participation_rate=0.5,
        )
        MarketMakerAS(
            model=self, trader_id=4, rng=np.random.default_rng(4),
            horizon=5000, kappa=50, gamma=0.1, sigma=0.05,
        )

        # ---- DataCollector ----
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "BestBid":        lambda m: m.book.best_bid(),
                "BestAsk":        lambda m: m.book.best_ask(),
                "Spread":         lambda m: m.book.spread(),
                "Mid":            lambda m: m.book.mid_price(),
                "OBI":            lambda m: m.book.book_imbalance(),
                "Microprice":     lambda m: microprice(m.book),
                "OBI_5":          lambda m: top_n_obi(m.book, 5),
                "BestBidDepth":   lambda m: best_level_depth(m.book)[0],
                "BestAskDepth":   lambda m: best_level_depth(m.book)[1],
                "Top5BidDepth":   lambda m: top_n_depth(m.book, 5)[0],
                "Top5AskDepth":   lambda m: top_n_depth(m.book, 5)[1],
                "TotalBidDepth":  lambda m: total_visible_depth(m.book)[0],
                "TotalAskDepth":  lambda m: total_visible_depth(m.book)[1],
                "NumBidLevels":   lambda m: len(m.book.bid_prices),
                "NumAskLevels":   lambda m: len(m.book.ask_prices),
                "MMAS_Inventory": lambda m: m._get_mmas_inventory(),
                "DPBidDepth":     lambda m: m.dark_pool.queue_depth()[0],
                "DPAskDepth":     lambda m: m.dark_pool.queue_depth()[1],
                "DPRecentVolume": lambda m: m.dark_pool.recent_volume(m.current_step, 20),
                "DPPendingRoutes": lambda m: len(m.dark_pool.pending_lit_routes),
            },
        )

        # Per-step tracking
        self.order_records = []
        self.trade_records = []
        self.dp_trade_records = []
        self.snapshots = []
        self._trades_this_step = []

    # ------------------------------------------------------------------
    def _get_mmas_agent(self):
        for agent in self.agents:
            if isinstance(agent, MarketMakerAS):
                return agent
        return None

    def _get_mmas_inventory(self):
        a = self._get_mmas_agent()
        return a.inventory if a is not None else 0

    # ------------------------------------------------------------------
    def step(self):
        if self.current_step >= self.steps_to_run:
            self.running = False
            return
        t = self.current_step
        self._trades_this_step = []
        inventory_changes = 0

        # Advance fundamental
        self.fundamental.step()

        # Sequential agent activation (same loop as original run_simulation)
        for agent in self.agents:
            # MarketMaker gets multiple actions per step
            if isinstance(agent, MarketMaker):
                n_actions = int(self._sim_rng.integers(1, 4))
            else:
                n_actions = 1

            for _ in range(n_actions):
                action = agent.act(t, self.book)
                print(action)
                trades = apply_action(self.book, action)

                # Record orders
                if isinstance(action, Order):
                    self.order_records.append({
                        "t": t, "Agent": agent.trader_id,
                        "OrderType": "market" if action.price is None else "limit",
                        "Price": action.price, "Qty": action.qty, "Side": action.side,
                    })

                # Record trades
                for tr in trades:
                    self.trade_records.append({
                        "t": tr.ts, "Price": tr.price, "Qty": tr.qty,
                        "AggressorSide": tr.aggressor_side,
                        "MakerOrderID": tr.maker_order_id,
                        "TakerOrderID": tr.taker_order_id,
                    })
                    self._trades_this_step.append(tr)

            # Institutional traders also submit to dark pool
            if isinstance(agent, InstitutionalTrader):
                dp_trades = agent.act_dark(t, self.dark_pool)
                if dp_trades:
                    for tr in dp_trades:
                        self.dp_trade_records.append({
                            "t": tr.timestamp, "Price": tr.price, "Qty": tr.qty,
                        })

        # Notify MarketMakerAS of fills
        mmAS_agent = self._get_mmas_agent()
        for trade in self._trades_this_step:
            if mmAS_agent is not None:
                if (
                    trade.maker_trader_id == mmAS_agent.trader_id
                    or trade.taker_trader_id == mmAS_agent.trader_id
                ):
                    mmAS_agent.update_inventory(trade)
                    inventory_changes += 1

                    if (
                        trade.maker_order_id == mmAS_agent.last_bid_id
                        or trade.taker_order_id == mmAS_agent.last_bid_id
                    ):
                        mmAS_agent.last_bid_id = None
                    elif (
                        trade.maker_order_id == mmAS_agent.last_ask_id
                        or trade.taker_order_id == mmAS_agent.last_ask_id
                    ):
                        mmAS_agent.last_ask_id = None

        # Advance dark pool clock
        self.dark_pool.tick(t)

        # Sanity checks
        bb, ba = self.book.best_bid(), self.book.best_ask()
        if not np.isfinite(bb):
            print(f"WARNING: bid book empty at t={t}")
        if not np.isfinite(ba):
            print(f"WARNING: ask book empty at t={t}")
        if np.isfinite(bb) and np.isfinite(ba) and bb >= ba:
            raise ValueError(f"BOOK CROSSED: best_bid={bb} >= best_ask={ba}")

        print(f"Inventory changes for MMAS/Trades made: {inventory_changes}")

        # Collect data
        self.datacollector.collect(self)

        # Periodic logging and snapshots
        if t % 1 == 0:
            sp = self.book.spread()
            obi = self.book.book_imbalance()
            print(
                f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={sp:.4f}, obi={obi:.4f}, "
                f"bids : {self.book.top_n_levels('buy', 20)}, asks : {self.book.top_n_levels('sell', 20)}"
            )

        if t % 25 == 0:
            sp = self.book.spread()
            obi = self.book.book_imbalance()
            print(f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={sp:.4f}, obi={obi:.4f}")
            self.snapshots.append({
                "t": t,
                "bids": self.book.top_n_levels("buy", 10),
                "asks": self.book.top_n_levels("sell", 10),
            })

        self.current_step += 1

    # ------------------------------------------------------------------
    def run(self):
        for _ in range(self.steps_to_run):
            self.step()

    # ------------------------------------------------------------------
    def get_results(self):
        """Return all DataFrames matching the original run_simulation output."""
        book_df = self.datacollector.get_model_vars_dataframe().reset_index()
        book_df = book_df.rename(columns={"index": "t"})
        return (
            pd.DataFrame(self.order_records),
            book_df,
            pd.DataFrame(self.trade_records),
            pd.DataFrame(self.dp_trade_records),
            self.snapshots,
        )

    # ------------------------------------------------------------------
    def save_plots(self, out_dir: str = None):
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "ABM_OB_plots")
        os.makedirs(out_dir, exist_ok=True)

        orders_df, book_df, trades_df, dp_trades_df, snapshots = self.get_results()
        book_df = add_market_analytics(book_df, vol_window=10)

        plot_series(book_df["Spread"], "Spread", os.path.join(out_dir, "spread.png"))
        plot_series(book_df["Mid"], "Mid-price diffusion", os.path.join(out_dir, "mid.png"))
        plot_series(book_df["OBI"], "Orderbook Imbalance", os.path.join(out_dir, "obi.png"))
        plot_series(book_df["DPRecentVolume"], "Dark Pool: Recent Traded Volume (20-tick)",
                    os.path.join(out_dir, "dp_volume.png"))

        plot_mmAS_inventory(book_df, os.path.join(out_dir, "mmAS_inventory.png"))
        plot_mid_vs_fundamental(book_df, self.fundamental.history,
                                os.path.join(out_dir, "mid_vs_fundamental.png"))

        if snapshots:
            plot_snapshots(snapshots, os.path.join(out_dir, "ABM_OrderBook_Snapshots.png"))

        final_bids = self.book.top_n_levels("buy", 20)
        final_asks = self.book.top_n_levels("sell", 20)
        plot_demand_curve(final_bids, final_asks, os.path.join(out_dir, "demand_curve.png"))

        orders_df.to_csv(os.path.join(out_dir, "orders_log.csv"), index=False)
        book_df.to_csv(os.path.join(out_dir, "book_log.csv"), index=False)
        trades_df.to_csv(os.path.join(out_dir, "trades_log.csv"), index=False)
        dp_trades_df.to_csv(os.path.join(out_dir, "dp_trades_log.csv"), index=False)

        print(f"Results saved to {out_dir}/")