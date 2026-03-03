import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .orderbook import OrderBook, Order
from .agents import NoiseTrader, MarketMaker, InstitutionalTrader, Action


def apply_action(book: OrderBook, action: Action, t: int) -> int:
    """
    Applies an agent action to the order book.
    Returns number of trades executed (for basic debugging).
    """
    if action is None:
        return 0

    if isinstance(action, tuple) and action[0] == "cancel":
        book.cancel(action[1])
        return 0

    if isinstance(action, Order):
        if action.price is None:
            trades = book.execute_market(action)
        else:
            trades = book.add_limit(action)
        return len(trades)

    raise TypeError(f"Unknown action type: {type(action)}")


def seed_initial_book(book: OrderBook, best_bid: float = 100.0, best_ask: float = 101.0, levels: int = 5, rng=None):
    rng = rng or np.random.default_rng(42)
    # bids
    for i in range(levels):
        px = best_bid - book.tick * i
        qty = int(rng.integers(1, 10))
        book.add_limit_post_only(Order(order_id=10_000 + i, trader_id=-1, side="buy", qty=qty, price=px, ts=0))
    # asks
    for i in range(levels):
        px = best_ask + book.tick * i
        qty = int(rng.integers(1, 10))
        book.add_limit_post_only(Order(order_id=20_000 + i, trader_id=-1, side="sell", qty=qty, price=px, ts=0))


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "ABM_OB_plots")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    book = OrderBook(tick=0.01, max_depth_levels=10)
    seed_initial_book(book, best_bid=100.0, best_ask=101.0, levels=5, rng=rng)

    agents = [
        NoiseTrader(trader_id=1, rng=np.random.default_rng(1)),
        MarketMaker(trader_id=2, rng=np.random.default_rng(2)),
        InstitutionalTrader(trader_id=3, rng=np.random.default_rng(3)),
    ]

    steps = 200

    logs = {
        "t": [],
        "BestBid": [],
        "BestAsk": [],
        "Spread": [],
        "Mid": [],
        "Trades": [],
        "Orders": [],
        "BidLvls": [],
        "AskLvls": [],
        "LastActor": [],
    }

    snapshots = []

    for t in range(steps):
        trades_this_t = 0
        last_actor = None

        # simple sequential activation
        for a in agents:
            action = a.act(t, book)
            print(action)
            ntr = apply_action(book, action, t)
            if action is not None:
                last_actor = a.__class__.__name__
            trades_this_t += ntr
            #print(trades_this_t)

        bb, ba = book.best_bid(), book.best_ask()
        best_bid_qty = sum(o.qty for o in book.bids.get(bb, [])) if bb > 0 else 0
        best_ask_qty = sum(o.qty for o in book.asks.get(ba, [])) if np.isfinite(ba) else 0
        #print("best_bid_qty", best_bid_qty, "best_ask_qty", best_ask_qty)
        sp = book.spread()
        md = book.mid_price()

        logs["t"].append(t)
        logs["BestBid"].append(bb)
        logs["BestAsk"].append(ba)
        logs["Spread"].append(sp)
        logs["Mid"].append(md)
        logs["Trades"].append(trades_this_t)
        logs["Orders"].append(len(book.order_index))
        logs["BidLvls"].append(len(book.bid_prices))
        logs["AskLvls"].append(len(book.ask_prices))
        logs["LastActor"].append(last_actor)

        if t % 50 == 0:
            print(f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={sp:.4f}")

            snapshots.append({
                "t": t,
                "bids": book.top_n_levels("buy", 10),
                "asks": book.top_n_levels("sell", 10),
                "last_actor": last_actor
            })

    df = pd.DataFrame(logs)

    # Plots (optional)
    plt.figure(figsize=(12, 4))
    plt.plot(df["Spread"], label="Spread")
    plt.title("Spread")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spread.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(df["Mid"], label="Mid")
    plt.title("Mid-price diffusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mid.png"))
    plt.close()

    # Snapshot viz (simple)
    if snapshots:
        fig, axes = plt.subplots(len(snapshots), 1, figsize=(18, 2.5 * len(snapshots)))
        if len(snapshots) == 1:
            axes = [axes]

        for ax, snap in zip(axes, snapshots):
            bids = snap["bids"]
            asks = snap["asks"]

            bid_prices = [p for p, _ in bids]
            bid_qty = [q for _, q in bids]
            ask_prices = [p for p, _ in asks]
            ask_qty = [q for _, q in asks]

            ax.bar(bid_prices, bid_qty, width=0.002, label="Bids")
            ax.bar(ask_prices, [-q for q in ask_qty], width=0.002, label="Asks")
            ax.axhline(0, linewidth=1)
            ax.set_title(f"t={snap['t']} | last_actor={snap['last_actor']}")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ABM_OrderBook_Snapshots.png"))
        plt.close()

    # Save logs
    df.to_csv(os.path.join(out_dir, "run_log.csv"), index=False)


if __name__ == "__main__":
    main()