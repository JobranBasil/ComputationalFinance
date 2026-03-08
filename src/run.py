import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .orderbook import OrderBook, Order
from .agents import NoiseTrader, MarketMaker, InstitutionalTrader, Action


# ---------------------------------------------------------------------------
# Order book helpers
# ---------------------------------------------------------------------------

def apply_action(book: OrderBook, action: Action) -> int:
    """Apply an agent action to the order book. Returns number of trades executed."""
    if action is None:
        return 0
    if isinstance(action, tuple) and action[0] == "cancel":
        book.cancel(action[1])
        return 0
    if isinstance(action, Order):
        trades = book.execute_market(action) if action.price is None else book.add_limit(action)
        return len(trades)
    raise TypeError(f"Unknown action type: {type(action)}")


def seed_initial_book(
    book: OrderBook,
    best_bid: float = 100.0,
    best_ask: float = 101.0,
    levels: int = 5,
    rng: np.random.Generator | None = None,
) -> None:
    """Seed the order book with synthetic resting liquidity around an initial spread."""
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

    # Fixed half-width in price units across all snapshots
    tick = snapshots[0]["bids"][0][0] - snapshots[0]["bids"][1][0]
    bar_width = tick * 0.8
    half_width = tick * 7  # controls how many levels are visible — adjust as needed

    for ax, snap in zip(axes, snapshots):
        bid_prices, bid_qtys = zip(*snap["bids"])
        ask_prices, ask_qtys = zip(*snap["asks"])
        mid = (ask_prices[0] + bid_prices[0]) / 2

        ax.bar(bid_prices, bid_qtys, width=bar_width, label="Bids")
        ax.bar(ask_prices, [-q for q in ask_qtys], width=bar_width, label="Asks")
        ax.axhline(0, linewidth=1)
        ax.axvline(mid, linewidth=1, color="black", linestyle="--")
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


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(book: OrderBook, agents: list, steps: int) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Run the ABM simulation.
    Returns order-level logs, book-state logs, and periodic depth snapshots.
    """
    order_records = []
    book_records = []
    snapshots = []

    for t in range(steps):
        print(f"TIME STEP : {t}")

        for agent in agents:
            action = agent.act(t, book)
            apply_action(book, action)
            print(action)

            if isinstance(action, Order) and action.qty > 0:
                order_records.append({
                    "t": t,
                    "Agent": agent.trader_id,
                    "Order": type(action).__name__,
                    "Price": action.price,
                    "Qty": action.qty,
                    "Side": action.side,
                })

        bb, ba = book.best_bid(), book.best_ask()
        if bb >= ba:
            raise ValueError(f"BOOK CROSSED: best_bid={bb} >= best_ask={ba}")

        spread = ba - bb
        mid = (bb + ba) / 2
        obi = book.book_imbalance()
        bid_depth = book.top_n_levels("buy", 10)
        ask_depth = book.top_n_levels("sell", 10)

        book_records.append({
            "t": t, "BestBid": bb, "BestAsk": ba,
            "Spread": spread, "Mid": mid,
            "BidLvls": bid_depth, "AskLvls": ask_depth, "OBI": obi,
        })

        print(f"spread={spread:.4f} | bids: {bid_depth} | asks: {ask_depth}")

        if t % 50 == 0:
            print(f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={spread:.4f}, obi={obi:.4f}")
            snapshots.append({"t": t, "bids": bid_depth, "asks": ask_depth})

    return pd.DataFrame(order_records), pd.DataFrame(book_records), snapshots


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:

    #Connect directory to ABM_OB_plots
    out_dir = os.path.join(os.path.dirname(__file__), "ABM_OB_plots")
    os.makedirs(out_dir, exist_ok=True)

    #Seed the order book with some initial liquidity and create agents with distinct RNGs for independent behavior
    rng = np.random.default_rng(42)
    book = OrderBook(tick=0.01, max_depth_levels=10)
    seed_initial_book(book, best_bid=100.05, best_ask=100.1, levels=5, rng=rng)

    # Each NoiseTrader gets a distinct seed so they behave independently
    agents = [
        *[NoiseTrader(trader_id=i, rng=np.random.default_rng(i)) for i in range(1, 8)],
        MarketMaker(trader_id=8, rng=np.random.default_rng(8)),
        InstitutionalTrader(trader_id=9, rng=np.random.default_rng(9)),
        InstitutionalTrader(trader_id=10, rng=np.random.default_rng(10)),
    ]

    #Get the order and book logs, and periodic snapshots of the order book state for visualization
    orders_df, book_df, snapshots = run_simulation(book, agents, steps=2500)

    #Plot each time series and save the order book snapshots and demand curve at the end of the simulation
    plot_series(book_df["Spread"], "Spread",              os.path.join(out_dir, "spread.png"))
    plot_series(book_df["Mid"],    "Mid-price diffusion", os.path.join(out_dir, "mid.png"))
    plot_series(book_df["OBI"],    "Orderbook Imbalance", os.path.join(out_dir, "obi.png"))

    if snapshots:
        plot_snapshots(snapshots, os.path.join(out_dir, "ABM_OrderBook_Snapshots.png"))

    #Get the demand curve at the end of the simulation and save it
    final_bids = book.top_n_levels("buy", 10)
    final_asks = book.top_n_levels("sell", 10)
    plot_demand_curve(final_bids, final_asks, os.path.join(out_dir, "demand_curve.png"))

    #Save the order and book logs as CSV files for further analysis
    orders_df.to_csv(os.path.join(out_dir, "orders_log.csv"), index=False)
    book_df.to_csv(os.path.join(out_dir, "book_log.csv"), index=False)

if __name__ == "__main__":
    main()