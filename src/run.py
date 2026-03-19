import os
import sys
import importlib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .fundemental import FundamentalProcess
from .orderbook import OrderBook, Order
from .agents import (
    NoiseTrader,
    MarketMaker,
    InstitutionalTrader,
    MarketMakerAS,
    InformedTrader,
    Action,
)

_dp_mod = importlib.import_module("src.dark_pool")
DarkPool = _dp_mod.DarkPool
DarkOrder = _dp_mod.Order


# ---------------------------------------------------------------------------
# Order book helpers
# ---------------------------------------------------------------------------

def apply_action(book: OrderBook, action: Action | list) -> list:
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


def seed_initial_book(
    book: OrderBook,
    best_bid: float = 100.0,
    best_ask: float = 101.0,
    levels: int = 10,
    rng: np.random.Generator | None = None,
) -> None:
    """Seed the order book with synthetic resting liquidity around an initial spread."""
    rng = rng or np.random.default_rng(42)
    for i in range(levels):
        book.add_limit_post_only(Order(
            order_id=10_000 + i,
            trader_id=-1,
            side="buy",
            qty=int(rng.integers(1, 10)),
            price=best_bid - book.tick * i,
            ts=0,
        ))
        book.add_limit_post_only(Order(
            order_id=20_000 + i,
            trader_id=-1,
            side="sell",
            qty=int(rng.integers(1, 10)),
            price=best_ask + book.tick * i,
            ts=0,
        ))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def best_level_depth(book: OrderBook) -> tuple[int, int]:
    bb, ba = book.best_bid(), book.best_ask()
    bid_qty = sum(o.qty for o in book.bids.get(bb, [])) if np.isfinite(bb) else 0
    ask_qty = sum(o.qty for o in book.asks.get(ba, [])) if np.isfinite(ba) else 0
    return bid_qty, ask_qty


def total_visible_depth(book: OrderBook) -> tuple[int, int]:
    bid_total = sum(o.qty for p in book.bid_prices for o in book.bids[p])
    ask_total = sum(o.qty for p in book.ask_prices for o in book.asks[p])
    return bid_total, ask_total


def top_n_depth(book: OrderBook, n: int = 5) -> tuple[int, int]:
    bid_depth = sum(q for _, q in book.top_n_levels("buy", n))
    ask_depth = sum(q for _, q in book.top_n_levels("sell", n))
    return bid_depth, ask_depth


def top_n_obi(book: OrderBook, n: int = 5) -> float:
    bid_depth, ask_depth = top_n_depth(book, n)
    denom = bid_depth + ask_depth
    return (bid_depth - ask_depth) / denom if denom > 0 else 0.0


def microprice(book: OrderBook) -> float:
    bb, ba = book.best_bid(), book.best_ask()
    if not np.isfinite(bb) or not np.isfinite(ba):
        return np.nan

    qb = sum(o.qty for o in book.bids.get(bb, []))
    qa = sum(o.qty for o in book.asks.get(ba, []))
    denom = qb + qa
    if denom == 0:
        return np.nan

    return (ba * qb + bb * qa) / denom


def trade_volume(trades: list) -> int:
    return sum(tr.qty for tr in trades)


def signed_trade_volume(trades: list) -> int:
    return sum(tr.qty if tr.aggressor_side == "buy" else -tr.qty for tr in trades)


def vwap(trades: list) -> float:
    if not trades:
        return np.nan
    total_qty = sum(tr.qty for tr in trades)
    if total_qty == 0:
        return np.nan
    return sum(tr.price * tr.qty for tr in trades) / total_qty


def add_market_analytics(book_df: pd.DataFrame, vol_window: int = 10) -> pd.DataFrame:
    df = book_df.copy()

    df["MidReturn"] = df["Mid"].diff()
    df["LogMidReturn"] = np.log(df["Mid"] / df["Mid"].shift(1))
    df["RollingVol"] = df["LogMidReturn"].rolling(vol_window).std()
    df["SpreadChange"] = df["Spread"].diff()
    df["MicropriceReturn"] = df["Microprice"].diff()

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_series(series: pd.Series, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series, label=title)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_snapshots(snapshots: list[dict], out_path: str) -> None:
    fig, axes = plt.subplots(len(snapshots), 1, figsize=(8, 3 * len(snapshots)))
    if len(snapshots) == 1:
        axes = [axes]

    # safer tick handling
    tick = 0.01
    if snapshots and len(snapshots[0].get("bids", [])) >= 2:
        tick = abs(snapshots[0]["bids"][0][0] - snapshots[0]["bids"][1][0]) or 0.01

    bar_width = tick * 0.4
    half_width = tick * 22

    for ax, snap in zip(axes, snapshots):
        bids = snap["bids"]
        asks = snap["asks"]

        if bids:
            bid_prices, bid_qtys = zip(*bids)
            ax.bar(bid_prices, bid_qtys, width=bar_width, label="Bids")
        else:
            bid_prices, bid_qtys = [], []

        if asks:
            ask_prices, ask_qtys = zip(*asks)
            ax.bar(ask_prices, [-q for q in ask_qtys], width=bar_width, label="Asks")
        else:
            ask_prices, ask_qtys = [], []

        ax.axhline(0, linewidth=1)

        if bids and asks:
            mid = (ask_prices[0] + bid_prices[0]) / 2
            ax.axvline(mid, linewidth=1, color="black", linestyle="--")
            ax.set_xlim(mid - half_width, mid + half_width)
        elif bids:
            mid = bid_prices[0]
            ax.set_xlim(mid - half_width, mid + half_width)
        elif asks:
            mid = ask_prices[0]
            ax.set_xlim(mid - half_width, mid + half_width)

        ax.set_title(f"t={snap['t']}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")
        ax.ticklabel_format(useOffset=False, style="plain")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_demand_curve(bids: list[tuple], asks: list[tuple], out_path: str) -> None:
    bid_prices = [p for p, _ in bids]
    ask_prices = [p for p, _ in asks]
    bid_cum_qty = [sum(q for p, q in bids if p >= price) for price in bid_prices]
    ask_cum_qty = [sum(q for p, q in asks if p <= price) for price in ask_prices]

    fig, ax = plt.subplots()
    ax.step(bid_cum_qty, bid_prices, where="post", color="blue", label="Bids")
    ax.step(ask_cum_qty, ask_prices, where="post", color="red", label="Asks")
    ax.set_xlabel("Cumulative Quantity")
    ax.set_ylabel("Price")
    ax.set_title("Demand Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mmAS_inventory(book_df: pd.DataFrame, out_path: str) -> None:
    """Plot MarketMakerAS inventory over simulation time."""
    if "MMAS_Inventory" not in book_df.columns:
        print("No MMAS_Inventory column found in book_df — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(book_df["t"], book_df["MMAS_Inventory"], label="MMAS Inventory", color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Zero inventory")
    ax.fill_between(
        book_df["t"],
        book_df["MMAS_Inventory"],
        0,
        where=book_df["MMAS_Inventory"] > 0,
        alpha=0.2,
        color="green",
        label="Long",
    )
    ax.fill_between(
        book_df["t"],
        book_df["MMAS_Inventory"],
        0,
        where=book_df["MMAS_Inventory"] < 0,
        alpha=0.2,
        color="red",
        label="Short",
    )
    ax.set_title("MarketMakerAS Inventory Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Inventory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mid_vs_fundamental(
    book_df: pd.DataFrame,
    fundamental_history: list[float],
    out_path: str,
) -> None:
    """Plot mid-price vs fundamental value over simulation time."""
    fig, ax = plt.subplots(figsize=(14, 5))

    fund_series = fundamental_history[:len(book_df)]

    ax.plot(book_df["t"], book_df["Mid"], label="Mid Price", color="steelblue", linewidth=1.2)
    ax.plot(
        book_df["t"],
        fund_series,
        label="Fundamental Value",
        color="crimson",
        linewidth=1.2,
        linestyle="--",
    )

    ax.fill_between(
        book_df["t"],
        book_df["Mid"],
        fund_series,
        alpha=0.15,
        color="purple",
        label="Mispricing",
    )

    ax.set_title("Mid Price vs Fundamental Value")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    book: OrderBook,
    agents: list,
    dark_pool,
    steps: int,
    fundamental=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    order_records = []
    book_records = []
    trade_records = []
    dp_trade_records = []
    snapshots = []
    rng = np.random.default_rng(42)

    inventory_changes = 0

    for t in range(steps):
        print(f"\nTIME STEP : {t}\n")

        if fundamental is not None:
            fundamental.step()

        trades_this_step = []

        for agent in agents:
            if isinstance(agent, MarketMaker):
                n_actions = int(rng.integers(1, 4))
            else:
                n_actions = 1

            for _ in range(n_actions):
                action = agent.act(t, book)
                print(action)
                trades = apply_action(book, action)

                if isinstance(action, Order):
                    order_records.append({
                        "t": t,
                        "Agent": agent.trader_id,
                        "OrderType": "market" if action.price is None else "limit",
                        "Price": action.price,
                        "Qty": action.qty,
                        "Side": action.side,
                    })

                for tr in trades:
                    trade_records.append({
                        "t": tr.ts,
                        "Price": tr.price,
                        "Qty": tr.qty,
                        "AggressorSide": tr.aggressor_side,
                        "MakerOrderID": tr.maker_order_id,
                        "TakerOrderID": tr.taker_order_id,
                    })
                    trades_this_step.append(tr)

            # institutional traders also submit to the dark pool
            if isinstance(agent, InstitutionalTrader) or isinstance(agent, InformedTrader):
                dp_trades = agent.act_dark(t, dark_pool)
                if dp_trades:
                    for tr in dp_trades:
                        dp_trade_records.append({
                            "t": tr.timestamp,
                            "Price": tr.price,
                            "Qty": tr.qty,
                        })

        # Notify MarketMakerAS of fills
        mmAS_agent = next((agent for agent in agents if isinstance(agent, MarketMakerAS)), None)
        for trade in trades_this_step:
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

        # advance dark pool clock, expiry and routing fire here
        dark_pool.tick(t)

        bb, ba = book.best_bid(), book.best_ask()
        sp = book.spread()
        md = book.mid_price()
        obi = book.book_imbalance()

        bid_best_depth, ask_best_depth = best_level_depth(book)
        bid_total_depth, ask_total_depth = total_visible_depth(book)
        bid_depth_5, ask_depth_5 = top_n_depth(book, 5)
        obi_5 = top_n_obi(book, 5)
        mp = microprice(book)

        step_trade_count = len(trades_this_step)
        step_trade_volume = trade_volume(trades_this_step)
        step_signed_volume = signed_trade_volume(trades_this_step)
        step_vwap = vwap(trades_this_step)

        print(f"Inventory changes for MMAS/Trades made: {inventory_changes}")

        if not np.isfinite(bb):
            print(f"WARNING: bid book empty at t={t}")
        if not np.isfinite(ba):
            print(f"WARNING: ask book empty at t={t}")

        if np.isfinite(bb) and np.isfinite(ba) and bb >= ba:
            raise ValueError(f"BOOK CROSSED: best_bid={bb} >= best_ask={ba}")

        dp_bid_depth, dp_ask_depth = dark_pool.queue_depth()
        dp_recent_vol = dark_pool.recent_volume(current_ts=t, lookback=20)

        book_records.append({
            "t": t,
            "BestBid": bb,
            "BestAsk": ba,
            "Spread": sp,
            "RelativeSpread": sp / md if np.isfinite(md) and md != 0 else np.nan,
            "Mid": md,
            "Microprice": mp,
            "MidMinusMicroprice": md - mp if np.isfinite(md) and np.isfinite(mp) else np.nan,
            "OBI": obi,
            "OBI_5": obi_5,
            "BestBidDepth": bid_best_depth,
            "BestAskDepth": ask_best_depth,
            "Top5BidDepth": bid_depth_5,
            "Top5AskDepth": ask_depth_5,
            "TotalBidDepth": bid_total_depth,
            "TotalAskDepth": ask_total_depth,
            "NumBidLevels": len(book.bid_prices),
            "NumAskLevels": len(book.ask_prices),
            "TradesThisStep": step_trade_count,
            "TradeVolume": step_trade_volume,
            "SignedTradeVolume": step_signed_volume,
            "VWAP": step_vwap,
            "MMAS_Inventory": mmAS_agent.inventory if mmAS_agent is not None else 0,
            "DPBidDepth": dp_bid_depth,
            "DPAskDepth": dp_ask_depth,
            "DPRecentVolume": dp_recent_vol,
            "DPPendingRoutes": len(dark_pool.pending_lit_routes),
        })

        if t % 1 == 0:
            print(
                f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={sp:.4f}, obi={obi:.4f}, "
                f"bids : {book.top_n_levels('buy', 20)}, asks : {book.top_n_levels('sell', 20)}"
            )

        if t % 25 == 0:
            print(f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={sp:.4f}, obi={obi:.4f}")
            snapshots.append({
                "t": t,
                "bids": book.top_n_levels("buy", 10),
                "asks": book.top_n_levels("sell", 10),
            })

    return (
        pd.DataFrame(order_records),
        pd.DataFrame(book_records),
        pd.DataFrame(trade_records),
        pd.DataFrame(dp_trade_records),
        snapshots,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    fundamental = FundamentalProcess(
        start=100.075,
        sigma_v=0.03,
        rng=np.random.default_rng(99),
    )

    out_dir = os.path.join(os.path.dirname(__file__), "ABM_OB_plots")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    book = OrderBook(tick=0.01, max_depth_levels=20)
    seed_initial_book(book, best_bid=100.05, best_ask=100.10, levels=10, rng=rng)

    agents = [
        NoiseTrader(trader_id=1, rng=np.random.default_rng(1)),
        NoiseTrader(trader_id=5, rng=np.random.default_rng(5)),
        NoiseTrader(trader_id=6, rng=np.random.default_rng(6)),
        MarketMaker(trader_id=2, rng=np.random.default_rng(2)),
        InstitutionalTrader(trader_id=3, rng=np.random.default_rng(3)),
        InstitutionalTrader(trader_id=8, rng=np.random.default_rng(8)),
        InformedTrader(
            trader_id=7,
            rng=np.random.default_rng(7),
            fundamental=fundamental,
            sigma_s=0.08,
            participation_rate=0.5,
        ),
        MarketMakerAS(
            trader_id=4,
            rng=np.random.default_rng(4),
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

    orders_df, book_df, trades_df, dp_trades_df, snapshots = run_simulation(
        book,
        agents,
        dark_pool,
        steps=1000,
        fundamental=fundamental,
    )

    book_df = add_market_analytics(book_df, vol_window=10)

    plot_series(book_df["Spread"], "Spread", os.path.join(out_dir, "spread.png"))
    plot_series(book_df["Mid"], "Mid-price diffusion", os.path.join(out_dir, "mid.png"))
    plot_series(book_df["OBI"], "Orderbook Imbalance", os.path.join(out_dir, "obi.png"))
    plot_series(book_df["VWAP"], "Volume-Weighted Average Price (VWAP)", os.path.join(out_dir, "vwap.png"))
    plot_series(
        book_df["DPRecentVolume"],
        "Dark Pool: Recent Traded Volume (20-tick)",
        os.path.join(out_dir, "dp_volume.png"),
    )

    plot_mmAS_inventory(book_df, os.path.join(out_dir, "mmAS_inventory.png"))
    plot_mid_vs_fundamental(
        book_df,
        fundamental.history,
        os.path.join(out_dir, "mid_vs_fundamental.png"),
    )

    if snapshots:
        plot_snapshots(snapshots, os.path.join(out_dir, "ABM_OrderBook_Snapshots.png"))

    final_bids = book.top_n_levels("buy", 20)
    final_asks = book.top_n_levels("sell", 20)
    plot_demand_curve(final_bids, final_asks, os.path.join(out_dir, "demand_curve.png"))

    orders_df.to_csv(os.path.join(out_dir, "orders_log.csv"), index=False)
    book_df.to_csv(os.path.join(out_dir, "book_log.csv"), index=False)
    trades_df.to_csv(os.path.join(out_dir, "trades_log.csv"), index=False)
    dp_trades_df.to_csv(os.path.join(out_dir, "dp_trades_log.csv"), index=False)


if __name__ == "__main__":
    main()