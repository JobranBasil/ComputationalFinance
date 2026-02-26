# src/run.py
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .orderbook import OrderBook, Order
from .agents import Agent, Action, NoiseTrader, MarketMaker, Institutional


class Scheduler:
    """
    Picks which agent(s) act each timestep.
    weights ~ intensities (Poisson-style interpretation
    """
    def __init__(self, rng: np.random.Generator, agents: List[Agent], weights: List[float]):
        assert len(agents) == len(weights)
        self.rng = rng
        self.agents = agents
        w = np.array(weights, dtype=float)
        self.probs = w / w.sum()

    def pick_k(self, k: int) -> List[Agent]:
        idx = self.rng.choice(len(self.agents), size=k, replace=True, p=self.probs)
        return [self.agents[i] for i in idx]


class ABMSimulator:
    def __init__(self, book: OrderBook, steps: int = 10_000, seed: int = 42):
        self.book = book
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self._next_order_id = 1

        self.agents: List[Agent] = []
        self.scheduler: Scheduler | None = None

        # logs
        self.mid = []
        self.wmid = []
        self.spread = []
        self.best_bid = []
        self.best_ask = []
        self.last_actor = []
        self.snapshots = []

    def _new_order_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    def set_agents(self, agents: List[Agent], weights: List[float]):
        self.agents = agents
        for a in self.agents:
            a.new_order_id = self._new_order_id
        self.scheduler = Scheduler(self.rng, agents, weights)

    def seed_initial_book(self, best_bid: float = 100.0, best_ask: float = 101.0, levels: int = 5):
        for i in range(levels):
            px = self.book.snap(best_bid - self.book.tick * i)
            qty = int(self.rng.integers(1, 10))
            self.book.add_limit_post_only(Order(self._new_order_id(), -1, "buy", qty, price=px, ts=0))
        for i in range(levels):
            px = self.book.snap(best_ask + self.book.tick * i)
            qty = int(self.rng.integers(1, 10))
            self.book.add_limit_post_only(Order(self._new_order_id(), -1, "sell", qty, price=px, ts=0))

    def replenish_if_thin(self, min_levels: int, t: int):
        while len(self.book.bid_prices) < min_levels:
            start_px = min(self.book.best_bid() - self.book.tick, self.book.best_ask() - self.book.tick)
            px = self.book._next_free_price("buy", start_px)
            qty = int(self.rng.integers(1, 10))
            self.book.add_limit_post_only(Order(self._new_order_id(), -1, "buy", qty, price=px, ts=t))

        while len(self.book.ask_prices) < min_levels:
            start_px = max(self.book.best_ask() + self.book.tick, self.book.best_bid() + self.book.tick)
            px = self.book._next_free_price("sell", start_px)
            qty = int(self.rng.integers(1, 10))
            self.book.add_limit_post_only(Order(self._new_order_id(), -1, "sell", qty, price=px, ts=t))

    def _apply_actions(self, actions: List[Action]):
        for act in actions:
            if isinstance(act, Order):
                if act.price is not None:
                    act.price = self.book.snap(act.price)
                if act.price is None:
                    self.book.execute_market(act)
                else:
                    self.book.add_limit(act)
            else:
                kind, oid, cqty = act
                if kind == "cancel":
                    self.book.cancel(int(oid), cancel_qty=cqty)

    def run(self, actions_per_step: int = 1, snapshot_every: int = 1000, min_levels: int = 5) -> pd.DataFrame:
        assert self.scheduler is not None, "Call set_agents() first."

        for t in range(self.steps):
            self.replenish_if_thin(min_levels=min_levels, t=t)

            chosen = self.scheduler.pick_k(actions_per_step)

            last = "None"
            for ag in chosen:
                actions = ag.step(t, self.book)
                self._apply_actions(actions)
                last = type(ag).__name__

            bb, ba = self.book.best_bid(), self.book.best_ask()
            self.best_bid.append(bb)
            self.best_ask.append(ba)
            self.spread.append(self.book.spread())
            self.mid.append(self.book.mid_price())
            self.wmid.append(self.book.weighted_mid())
            self.last_actor.append(last)

            if t % snapshot_every == 0:
                print(f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={self.spread[-1]:.4f}")
                self.snapshots.append({
                    "t": t,
                    "bids": self.book.top_n_levels("buy", 10),
                    "asks": self.book.top_n_levels("sell", 10),
                    "event": last
                })

        return pd.DataFrame({
            "Mid": self.mid,
            "WeightedMid": self.wmid,
            "Spread": self.spread,
            "BestBid": self.best_bid,
            "BestAsk": self.best_ask,
            "LastActor": self.last_actor,
        }).dropna()


def visualise_order_book_snapshots(snapshots, out_path):
    num = len(snapshots)
    fig, axes = plt.subplots(num, 1, figsize=(18, 2.5 * num))
    if num == 1:
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
        ax.set_title(f"t={snap['t']} | last_actor={snap['event']}")
        ax.axhline(0, linewidth=1)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "ABM_OB_plots")
    os.makedirs(output_dir, exist_ok=True)

    book = OrderBook(tick=0.01, max_depth_levels=10)
    sim = ABMSimulator(book, steps=20_000, seed=42)
    sim.seed_initial_book(best_bid=100.0, best_ask=101.0, levels=5)

    rng = np.random.default_rng(42)
    agents = [
        NoiseTrader(agent_id=1, rng=rng, p_act=0.35, max_qty=4),
        MarketMaker(agent_id=2, rng=rng, quote_size=5, levels=1),
        Institutional(agent_id=3, rng=rng, side="buy", parent_qty=300, display_qty=10, refresh_size=10),
    ]
    weights = [10.0, 5.0, 1.0]  # agent selection weights

    sim.set_agents(agents, weights)

    results = sim.run(actions_per_step=1, snapshot_every=1000, min_levels=5)

    visualise_order_book_snapshots(sim.snapshots[:10], os.path.join(output_dir, "ABM_OrderBook_Snapshots.png"))

    plt.figure(figsize=(10, 4))
    plt.plot(results["Mid"], label="Mid")
    plt.legend()
    plt.title("Mid-price diffusion")
    plt.savefig(os.path.join(output_dir, "mid.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(results["Spread"], label="Spread")
    plt.legend()
    plt.title("Spread")
    plt.savefig(os.path.join(output_dir, "spread.png"))
    plt.close()


if __name__ == "__main__":
    main()