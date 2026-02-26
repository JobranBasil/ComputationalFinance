from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Literal

import numpy as np

Side = Literal["buy", "sell"]


@dataclass
class Order:
    order_id: int
    trader_id: int
    side: Side
    qty: int
    price: Optional[float] = None  # None for market
    ts: int = 0                    # time priority (lower ts = earlier)


@dataclass
class Trade:
    ts: int
    price: float
    qty: int
    aggressor_side: Side
    maker_order_id: int
    taker_order_id: int


class OrderBook:
    """
    Continuous double auction with price-time priority.

    - Price levels stored as: price -> deque[Order] (FIFO)
    - Best bid = max bid price, Best ask = min ask price
    - add_limit() matches if marketable; remaining qty posts
    - execute_market() matches until filled or opposite empty
    - cancel() removes order by id (O(n) search at level)
    """

    def __init__(self, tick: float = 0.01, max_depth_levels: int = 10):
        self.tick = tick
        self.max_depth_levels = max_depth_levels

        self.bids: Dict[float, Deque[Order]] = {}
        self.asks: Dict[float, Deque[Order]] = {}

        # sorted ascending; best bid = last, best ask = first
        self.bid_prices: List[float] = []
        self.ask_prices: List[float] = []

        # order_id -> (side, price)
        self.order_index: Dict[int, Tuple[Side, float]] = {}

    # ---------- best quotes ----------

    def best_bid(self) -> float:
        return self.bid_prices[-1] if self.bid_prices else 0.0

    def best_ask(self) -> float:
        return self.ask_prices[0] if self.ask_prices else float("inf")

    def spread(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) if (bb > 0 and np.isfinite(ba)) else np.nan

    def mid_price(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        return 0.5 * (bb + ba) if (bb > 0 and np.isfinite(ba)) else np.nan

    # ---------- helpers ----------

    def _add_price_level(self, side: Side, price: float) -> None:
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

    def _remove_price_level_if_empty(self, side: Side, price: float) -> None:
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

    def _enforce_max_depth(self) -> None:
        # Keep only best N price levels (simple pruning)
        if len(self.bid_prices) > self.max_depth_levels:
            worst = self.bid_prices[:-self.max_depth_levels]
            for p in worst:
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

    # ---------- depth ----------

    def top_n_levels(self, side: Side, n: int) -> List[Tuple[float, int]]:
        """
        Return [(price, total_qty_at_price), ...] for top n price levels.
        """
        out: List[Tuple[float, int]] = []

        if side == "buy":
            prices = list(reversed(self.bid_prices[-n:]))
            book = self.bids
        else:
            prices = self.ask_prices[:n]
            book = self.asks

        for p in prices:
            total = sum(o.qty for o in book[p])
            out.append((p, total))
        return out

    # ---------- public ops ----------

    def add_limit(self, order: Order) -> List[Trade]:
        assert order.price is not None, "Limit order must have a price"
        trades = self._match(order)

        if order.qty > 0:
            self._add_price_level(order.side, order.price)
            if order.side == "buy":
                self.bids[order.price].append(order)
            else:
                self.asks[order.price].append(order)

            self.order_index[order.order_id] = (order.side, order.price)
            self._enforce_max_depth()

        return trades

    def add_limit_post_only(self, order: Order) -> None:
        assert order.price is not None
        self._add_price_level(order.side, order.price)

        if order.side == "buy":
            self.bids[order.price].append(order)
        else:
            self.asks[order.price].append(order)

        self.order_index[order.order_id] = (order.side, order.price)
        self._enforce_max_depth()

    def execute_market(self, order: Order) -> List[Trade]:
        assert order.price is None, "Market order must have price=None"
        return self._match(order)

    def cancel(self, order_id: int, cancel_qty: Optional[int] = None) -> bool:
        """
        Cancel by id. If cancel_qty is None -> cancel full remaining.
        """
        if order_id not in self.order_index:
            return False

        side, price = self.order_index[order_id]
        book = self.bids if side == "buy" else self.asks
        q = book.get(price)
        if q is None:
            return False

        for o in list(q):
            if o.order_id == order_id:
                if cancel_qty is None or cancel_qty >= o.qty:
                    q.remove(o)
                    del self.order_index[order_id]
                else:
                    o.qty -= cancel_qty
                self._remove_price_level_if_empty(side, price)
                return True

        return False

    # ---------- matching ----------

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
            fill = min(taker.qty, maker.qty)

            trades.append(
                Trade(
                    ts=taker.ts,
                    price=px,
                    qty=fill,
                    aggressor_side=taker.side,
                    maker_order_id=maker.order_id,
                    taker_order_id=taker.order_id,
                )
            )

            taker.qty -= fill
            maker.qty -= fill

            if maker.qty == 0:
                maker_q.popleft()
                self.order_index.pop(maker.order_id, None)

            # clean up empty level
            if taker.side == "buy":
                self._remove_price_level_if_empty("sell", px)
            else:
                self._remove_price_level_if_empty("buy", px)

        return trades