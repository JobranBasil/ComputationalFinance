import os
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import faulthandler, signal
faulthandler.register(signal.SIGQUIT)   # Ctrl+\ will dump stack


Side = Literal["buy", "sell"]
OrderType = Literal["limit", "market", "cancel"]

# -----------------------------
# Core data objects
# -----------------------------

@dataclass
class Order:
    order_id: int
    trader_id: int
    side: Side
    qty: int
    price: Optional[float] = None          # None for market
    ts: int = 0                            # time priority
    # Iceberg support (optional)
    display_qty: Optional[int] = None
    hidden_qty: int = 0
    refresh_size: int = 0
    refresh_count: int = 0

    def is_iceberg(self) -> bool:
        return (self.hidden_qty > 0) and (self.display_qty is not None) and (self.refresh_size > 0)


@dataclass
class Trade:
    ts: int
    price: float
    qty: int
    aggressor_side: Side
    maker_order_id: int
    taker_order_id: int


# -----------------------------
# OrderBook with price-time priority
# -----------------------------

class OrderBook:
    """
    Price-time priority:
      - per price level: FIFO queue of orders
      - best bid: max price
      - best ask: min price
    """
    def __init__(self, tick: float = 0.01, max_depth_levels: int = 10):
        self.tick = tick
        self.max_depth_levels = max_depth_levels

        # price -> deque[Order]
        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}

        # keep sorted lists of active prices for fast best bid/ask
        self.bid_prices: List[float] = []   # sorted ascending; best is last
        self.ask_prices: List[float] = []   # sorted ascending; best is first

        # order_id -> (side, price)
        self.order_index: Dict[int, Tuple[Side, float]] = {}

    # ---- utilities ----

    def _next_free_price(self, side: Side, start_px: float) -> float:
        """
        Returns the first price on the tick grid >= start_px (asks) or <= start_px (bids)
        that is NOT already an existing price level.
        """
        # snap to tick grid to avoid float weirdness
        px = round(start_px / self.tick) * self.tick

        if side == "sell":
            while px in self.asks:
                px = round((px + self.tick) / self.tick) * self.tick
        else:  # buy
            while px in self.bids:
                px = round((px - self.tick) / self.tick) * self.tick

        return px

    def _add_price_level(self, side: Side, price: float):
        if side == "buy":
            if price not in self.bids:
                self.bids[price] = deque()
                self.bid_prices.append(price)
                self.bid_prices.sort()
        else:
            if price not in self.asks:
                self.asks[price] = deque()
                self.ask_prices.append(price)
                self.ask_prices.sort()

    def _remove_price_level_if_empty(self, side: Side, price: float):
        if side == "buy":
            q = self.bids.get(price)
            if q is not None and len(q) == 0:
                del self.bids[price]
                self.bid_prices.remove(price)
        else:
            q = self.asks.get(price)
            if q is not None and len(q) == 0:
                del self.asks[price]
                self.ask_prices.remove(price)

    def best_bid(self) -> float:
        return self.bid_prices[-1] if self.bid_prices else 0.0

    def best_ask(self) -> float:
        return self.ask_prices[0] if self.ask_prices else float("inf")

    def mid_price(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2 if (bb > 0 and np.isfinite(ba)) else np.nan

    def spread(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) if (bb > 0 and np.isfinite(ba)) else np.nan

    # ---- depth queries ----

    def top_n_levels(self, side: Side, n: int) -> List[Tuple[float, int]]:
        """
        Returns [(price, total_display_qty_at_price), ...] for top n price levels.
        """
        out = []
        if side == "buy":
            prices = list(reversed(self.bid_prices[-n:]))
            book = self.bids
        else:
            prices = self.ask_prices[:n]
            book = self.asks

        for p in prices:
            total = 0
            for o in book[p]:
                # displayed qty handling for icebergs
                if o.display_qty is None:
                    total += o.qty
                else:
                    total += min(o.qty, o.display_qty)
            out.append((p, total))
        return out

    def weighted_mid(self) -> float:
        bid_lvls = self.top_n_levels("buy", n=min(5, len(self.bid_prices)))
        ask_lvls = self.top_n_levels("sell", n=min(5, len(self.ask_prices)))

        bid_vol = sum(q for _, q in bid_lvls)
        ask_vol = sum(q for _, q in ask_lvls)
        if bid_vol <= 0 or ask_vol <= 0:
            return np.nan

        wbid = sum(p * q for p, q in bid_lvls) / bid_vol
        wask = sum(p * q for p, q in ask_lvls) / ask_vol
        return 0.5 * (wbid + wask)

    # ---- core operations ----

    def add_limit(self, order: Order) -> List[Trade]:
        """
        Add limit order. If crossing, match immediately (continuous double auction).
        Returns trades.
        """
        assert order.price is not None, "Limit order must have a price"
        trades: List[Trade] = []

        # First attempt to match if marketable
        trades += self._match(order)

        # If residual qty remains, post to book
        if order.qty > 0:
            self._add_price_level("buy" if order.side == "buy" else "sell", order.price)
            if order.side == "buy":
                self.bids[order.price].append(order)
            else:
                self.asks[order.price].append(order)
            self.order_index[order.order_id] = (order.side, order.price)

            # Optional: enforce max depth levels (by removing worst levels)
            self._enforce_max_depth()

        return trades
    
    def add_limit_post_only(self, order: Order) -> None:
        """Insert a limit order without matching (used for replenishment)."""
        assert order.price is not None

        self._add_price_level("buy" if order.side == "buy" else "sell", order.price)

        if order.side == "buy":
            self.bids[order.price].append(order)
        else:
            self.asks[order.price].append(order)

        self.order_index[order.order_id] = (order.side, order.price)
        self._enforce_max_depth()

    def execute_market(self, order: Order) -> List[Trade]:
        """
        Market order: match against opposite book until filled or book empty.
        Returns trades.
        """
        assert order.price is None, "Market order should have price=None"
        return self._match(order)

    def cancel(self, order_id: int, cancel_qty: Optional[int] = None) -> bool:
        """
        Cancel existing order by id. If cancel_qty is None -> cancel full remaining.
        Returns True if found and cancelled (fully or partially).
        """
        if order_id not in self.order_index:
            return False

        side, price = self.order_index[order_id]
        book = self.bids if side == "buy" else self.asks
        q = book.get(price)
        if q is None:
            return False

        # Find order inside FIFO queue (O(n) at price level; fine for coursework)
        for i in range(len(q)):
            o = q[i]
            if o.order_id == order_id:
                if cancel_qty is None or cancel_qty >= o.qty:
                    # remove order fully
                    q.remove(o)
                    del self.order_index[order_id]
                else:
                    o.qty -= cancel_qty
                self._remove_price_level_if_empty(side, price)
                return True
        return False

    # ---- internal matching ----

    def _match(self, taker: Order) -> List[Trade]:
        trades: List[Trade] = []

        def can_cross() -> bool:
            if taker.side == "buy":
                if not self.ask_prices:
                    return False
                best = self.ask_prices[0]
                return (taker.price is None) or (taker.price >= best)
            else:
                if not self.bid_prices:
                    return False
                best = self.bid_prices[-1]
                return (taker.price is None) or (taker.price <= best)

        while taker.qty > 0 and can_cross():
            if taker.side == "buy":
                px = self.ask_prices[0]
                maker_q = self.asks[px]
            else:
                px = self.bid_prices[-1]
                maker_q = self.bids[px]

            maker = maker_q[0]

            # displayed qty (iceberg): only that portion is visible/available at a time
            maker_available = maker.qty if maker.display_qty is None else min(maker.qty, maker.display_qty)
            fill = min(taker.qty, maker_available)

            # record trade
            trades.append(Trade(
                ts=taker.ts,
                price=px,
                qty=fill,
                aggressor_side=taker.side,
                maker_order_id=maker.order_id,
                taker_order_id=taker.order_id
            ))

            taker.qty -= fill
            maker.qty -= fill

            # iceberg refresh: if displayed portion depleted but hidden exists, refresh
            if maker.display_qty is not None:
                # if maker qty still > 0, that means there is remaining behind the display
                # but we model "display window" by refreshing when displayed is exhausted.
                # easiest: when maker.qty > 0 but we fully consumed visible window => refresh_count++
                # We'll approximate: if fill == maker_available and maker.qty > 0 and maker.hidden_qty > 0
                if fill == maker_available and maker.hidden_qty > 0:
                    refill = min(maker.refresh_size, maker.hidden_qty)
                    maker.hidden_qty -= refill
                    maker.qty += refill
                    maker.refresh_count += 1

            # remove maker order if fully done
            if maker.qty == 0:
                maker_q.popleft()
                if maker.order_id in self.order_index:
                    del self.order_index[maker.order_id]

            # remove empty price levels
            if taker.side == "buy":
                self._remove_price_level_if_empty("sell", px)
            else:
                self._remove_price_level_if_empty("buy", px)

        return trades

    def _enforce_max_depth(self):
        # Keep only best N price levels on each side (coarse but ok for coursework)
        if len(self.bid_prices) > self.max_depth_levels:
            worst = self.bid_prices[:-self.max_depth_levels]
            for p in worst:
                # delete all orders at that level (simplification)
                for o in self.bids[p]:
                    self.order_index.pop(o.order_id, None)
                del self.bids[p]
            self.bid_prices = self.bid_prices[-self.max_depth_levels:]

        if len(self.ask_prices) > self.max_depth_levels:
            worst = self.ask_prices[self.max_depth_levels:]
            for p in worst:
                for o in self.asks[p]:
                    self.order_index.pop(o.order_id, None)
                del self.asks[p]
            self.ask_prices = self.ask_prices[:self.max_depth_levels]


# -----------------------------
# Simulator (ZI baseline)
# -----------------------------

class ZISimulator:
    def __init__(
        self,
        book: OrderBook,
        steps: int = 100,
        lambda_market: float = 7,
        lambda_limit: float = 10,
        lambda_cancel: float = 3,
        seed: int = 42,
        tick: float = 0.01
    ):
        self.book = book
        self.steps = steps
        self.lambda_market = lambda_market
        self.lambda_limit = lambda_limit
        self.lambda_cancel = lambda_cancel
        self.rng = np.random.default_rng(seed)
        self.tick = tick

        self._next_order_id = 1

        # logs
        self.spread = []
        self.mid = []
        self.wmid = []
        self.event = []
        self.best_bid = []
        self.best_ask = []

        self.snapshots = []  # lightweight snapshots

    def _new_order_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    def seed_initial_book(self, best_bid: float = 100.0, best_ask: float = 101.0, levels: int = 5):
        # seed bids
        for i in range(levels):
            px = best_bid - self.tick * i
            qty = int(self.rng.integers(1, 10))
            o = Order(order_id=self._new_order_id(), trader_id=-1, side="buy", qty=qty, price=px, ts=0)
            self.book.add_limit_post_only(o)
        # seed asks
        for i in range(levels):
            px = best_ask + self.tick * i
            qty = int(self.rng.integers(1, 10))
            o = Order(order_id=self._new_order_id(), trader_id=-1, side="sell", qty=qty, price=px, ts=0)
            self.book.add_limit_post_only(o)

    def replenish_if_thin(self, min_levels: int = 5, t: int = 0):
        bb, ba = self.book.best_bid(), self.book.best_ask()

        # If one side empty, define an anchor so we can seed the other side safely
        if not np.isfinite(ba):
            ba = bb + 100 * self.tick  # arbitrary anchor above bid
        if bb <= 0:
            bb = ba - 100 * self.tick  # anchor below ask

        # Guard to prevent infinite loops no matter what
        guard = 0
        max_adds = 50  # plenty for min_levels=5

        while len(self.book.bid_prices) < min_levels:
            guard += 1
            if guard > max_adds:
                raise RuntimeError("replenish_if_thin guard hit on bids (no progress)")

            bb, ba = self.book.best_bid(), self.book.best_ask()
            if not np.isfinite(ba):
                ba = bb + 100 * self.tick
            if bb <= 0:
                bb = ba - 100 * self.tick

            start_px = min(bb - self.tick, ba - self.tick)
            px = self.book._next_free_price("buy", start_px)

            qty = int(self.rng.integers(1, 10))
            self.book.add_limit_post_only(Order(self._new_order_id(), -1, "buy", qty, price=px, ts=t))

        guard = 0
        while len(self.book.ask_prices) < min_levels:
            guard += 1
            if guard > max_adds:
                raise RuntimeError("replenish_if_thin guard hit on asks (no progress)")

            bb, ba = self.book.best_bid(), self.book.best_ask()
            if not np.isfinite(ba):
                ba = bb + 100 * self.tick
            if bb <= 0:
                bb = ba - 100 * self.tick

            # IMPORTANT: choose a NEW level based on how many ask levels currently exist
            start_px = max(ba + self.tick, bb + self.tick)
            px = self.book._next_free_price("sell", start_px)

            qty = int(self.rng.integers(1, 10))
            self.book.add_limit_post_only(Order(self._new_order_id(), -1, "sell", qty, price=px, ts=t))

    def run(self):
        weights = np.array([self.lambda_market, self.lambda_limit, self.lambda_cancel], dtype=float)
        probs = weights / weights.sum()

        for t in range(self.steps):
            self.replenish_if_thin(min_levels=5, t=t)

            event_type = self.rng.choice(["market", "limit", "cancel"], p=probs)

            if event_type == "market":
                qty = int(self.rng.integers(1, 5))
                side = "buy" if self.rng.random() < 0.5 else "sell"
                taker = Order(self._new_order_id(), trader_id=-2, side=side, qty=qty, price=None, ts=t)
                self.book.execute_market(taker)

            elif event_type == "limit":
                side = "buy" if self.rng.random() < 0.5 else "sell"
                qty = int(self.rng.integers(1, 10))
                shift = float(self.rng.uniform(0.005, 0.01))
                bb, ba = self.book.best_bid(), self.book.best_ask()
                px = (bb - shift) if side == "buy" else (ba + shift)

                maker = Order(self._new_order_id(), trader_id=-3, side=side, qty=qty, price=px, ts=t)
                self.book.add_limit(maker)

            else:  # cancel
                # cancel a random existing order (simple, ZI)
                if self.book.order_index:
                    oid = self.rng.choice(list(self.book.order_index.keys()))
                    self.book.cancel(int(oid), cancel_qty=None)

            # log
            bb, ba = self.book.best_bid(), self.book.best_ask()
            sp = self.book.spread()
            md = self.book.mid_price()
            wm = self.book.weighted_mid()

            self.best_bid.append(bb)
            self.best_ask.append(ba)
            self.spread.append(sp)
            self.mid.append(md)
            self.wmid.append(wm)
            self.event.append(event_type)

            if t % 1000 == 0:
                self.snapshots.append({
                    "t": t,
                    "bids": self.book.top_n_levels("buy", 10),
                    "asks": self.book.top_n_levels("sell", 10),
                    "event": event_type
                })

                print(f"t={t}, bb={bb:.4f}, ba={ba:.4f}, spread={sp:.4f}")
                print(f"t={t} ... bid_lvls={len(self.book.bid_prices)} ask_lvls={len(self.book.ask_prices)}")
        

        return pd.DataFrame({
            "Spread": self.spread,
            "Mid": self.mid,
            "WeightedMid": self.wmid,
            "Event": self.event,
            "BestBid": self.best_bid,
            "BestAsk": self.best_ask
        }).dropna()
    



# -----------------------------
# Plotting helpers (kept separate)
# -----------------------------

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

        ax.set_title(f"t={snap['t']} | event={snap['event']}")
        ax.axhline(0, linewidth=1)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Main (example run)
# -----------------------------

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "ZI_OB_plots")
    os.makedirs(output_dir, exist_ok=True)

    book = OrderBook(tick=0.01, max_depth_levels=10)
    sim = ZISimulator(book, steps=100000, lambda_market=7, lambda_limit=10, lambda_cancel=3, seed=42, tick=0.01)
    sim.seed_initial_book(best_bid=100.0, best_ask=101.0, levels=5)

    results = sim.run()

    # Snapshot plot
    visualise_order_book_snapshots(sim.snapshots[:100], os.path.join(output_dir, "Simulated_Order_Book_OOP.png"))

    # Time series plots
    plt.figure(figsize=(10, 4))
    plt.plot(results["Mid"], label="Mid")
    plt.title("Mid-price diffusion")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mid_price_diffusion_oop.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(results["Spread"], label="Spread")
    plt.title("Spread")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "spread_oop.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(results["Mid"], label="Mid", alpha=0.7)
    plt.plot(results["WeightedMid"], label="WeightedMid", alpha=0.7, linestyle="--")
    plt.title("Mid vs WeightedMid")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mid_vs_weightedmid_oop.png"))
    plt.close()