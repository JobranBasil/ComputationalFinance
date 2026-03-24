from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
import numpy as np
import math
import sys
import os
from .dark_pool import Order as DarkPoolOrder

from .fundemental import FundamentalProcess
from .orderbook import Order, OrderBook, Side, Trade

Action = Union[None, Order, Tuple[Literal["cancel"], int]]

# -------------------- NEW: Iceberg state (strategy-only) --------------------
@dataclass
class IcebergOrder:
    """
    Strategy-only iceberg:
    - 'remaining' is the hidden quantity left to execute (not shown in the book)
    - each visible slice is a normal LIMIT order with size=min(peak, remaining)
    - we consider a slice 'filled' when its order_id disappears from book.order_index
      (simple + minimal; does not require OrderBook edits)
    """
    side: Side
    remaining: int
    peak: int
    price: float

    active_order_id: Optional[int] = None
    active_slice_qty: int = 0  # the qty of the currently posted slice

# @dataclass
# class DarkIcebergOrder:
#     """
#
#     """
#     side: Side
#     remaining: int
#     peak: int
#     active_order_id: Optional[int] = None
#     active_slice_qty: int = 0


@dataclass
class BaseAgent:
    trader_id: int
    rng: np.random.Generator
    _next_order_id: int = 1

    # NEW: optional iceberg currently being executed by this agent
    iceberg: Optional[IcebergOrder] = None

    # track last valid mid so agents never snap to a hardcoded price
    _last_valid_mid: float = 100.0

    def new_oid(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        # make ids globally unique-ish by namespacing with trader_id
        return int(self.trader_id * 1_000_000 + oid)

    def _get_mid(self, book: OrderBook) -> float:
        """Return live mid if available, otherwise last known mid."""
        mid = book.mid_price()
        if np.isfinite(mid) and mid > 0:
            self._last_valid_mid = mid
            return mid
        return self._last_valid_mid

    def act(self, t: int, book: OrderBook) -> Action:
        return None

    # -------------------- NEW: iceberg helpers on BaseAgent --------------------
    def iceberg_start(self, side: Side, total_qty: int, peak: int, price: float) -> None:
        """
        Start a new iceberg parent order for this agent.
        The visible orders are standard LIMIT orders posted at 'price'.
        """
        self.iceberg = IcebergOrder(
            side=side,
            remaining=int(total_qty),
            peak=int(max(1, peak)),
            price=float(price),
            active_order_id=None,
            active_slice_qty=0,
        )

    def iceberg_step(self, t: int, book: OrderBook) -> Action:
        """
        Advance iceberg execution by at most ONE action this step:
        - If an active slice exists and is still in the book -> do nothing
        - If the active slice disappeared -> treat as filled and post the next slice
        - Repeat until remaining == 0, then clear iceberg

        IMPORTANT: This keeps everything minimal and only depends on book.order_index.
        """
        if self.iceberg is None:
            return None

        ice = self.iceberg

        # 1) If we have an active slice, check if it's still present in the book
        if ice.active_order_id is not None:
            if ice.active_order_id in book.order_index:
                # Still resting (not fully filled yet) -> no new action
                return None
            else:
                # Slice disappeared -> assume it got filled (or removed). Continue.
                ice.active_order_id = None
                ice.active_slice_qty = 0

        # 2) If no active slice and we still have remaining, post next slice
        if ice.remaining > 0:
            slice_qty = int(min(ice.peak, ice.remaining))
            oid = self.new_oid()

            ice.remaining -= slice_qty
            ice.active_order_id = oid
            ice.active_slice_qty = slice_qty

            return Order(
                oid,
                self.trader_id,
                ice.side,
                slice_qty,
                price=float(ice.price),
                ts=t,
            )

        # 3) Done: nothing remaining and no active slice
        self.iceberg = None
        return None


class NoiseTrader(BaseAgent):
    def __init__(
        self,
        trader_id: int,
        rng: np.random.Generator,
        participation_rate: float = 0.9,
        market_prob: float = 0.5,
        sign_persistence: float = 0.7,
        max_depth_ticks: int = 10,
    ):
        super().__init__(trader_id, rng)
        self.participation_rate = participation_rate
        self.market_prob = market_prob
        self.sign_persistence = sign_persistence
        self.max_depth_ticks = max_depth_ticks
        self.last_side: Optional[Side] = None

    def act(self, t: int, book: OrderBook):

        # participation decision
        if self.rng.random() > self.participation_rate:
            return None

        # persistent order sign
        if self.last_side is None or self.rng.random() > self.sign_persistence:
            side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        else:
            side = self.last_side

        self.last_side = side

        qty = int(self.rng.integers(10, 20))
        qty = int(self.rng.integers(10, 20))

        # market order
        if self.rng.random() < self.market_prob:
            return Order(self.new_oid(), self.trader_id, side, qty, price=None, ts=t)

        # limit order around mid
        mid = self._get_mid(book)

        tick = book.tick
        offset_ticks = int(self.rng.integers(0, self.max_depth_ticks + 1))

        if side == "buy":
            px = mid - offset_ticks * tick
        else:
            px = mid + offset_ticks * tick

        return Order(self.new_oid(), self.trader_id, side, qty, price=float(px), ts=t)


class MarketMakerAS(BaseAgent):
    """
    Market maker with inventory risk adjusted spread (Avellaneda-Stoikov).

    Key fix vs original: the MM now behaves like a designated market maker
    with a two-sided quoting obligation.
    """

    def __init__(self,
                 trader_id: int,
                 rng: np.random.Generator,
                 horizon: float,
                 kappa: float,
                 gamma: float,
                 sigma: float,
                 max_skew_ticks: int = 15,
                 inventory_limit: int = 200,
                 ):
        super().__init__(trader_id, rng)
        self.kappa = kappa
        self.gamma = gamma
        self.sigma = sigma
        self.T = horizon
        self.inventory = 0
        self.last_bid_id = None
        self.last_ask_id = None
        self.ttl = 50
        self.active_orders: dict = {}
        self.cancel = 0
        self.trades = 0
        self.max_skew_ticks = max_skew_ticks
        self.inventory_limit = inventory_limit
        self.last_valid_mid: float = 100.0

    def update_inventory(self, trade: Trade):
        """Update inventory based on executed trade."""
        if trade.maker_trader_id == self.trader_id:
            if trade.aggressor_side == "buy":
                self.inventory -= trade.qty
            else:
                self.inventory += trade.qty
        elif trade.taker_trader_id == self.trader_id:
            if trade.aggressor_side == "buy":
                self.inventory += trade.qty
            else:
                self.inventory -= trade.qty

    def optimal_spread(self, time_remaining: float) -> float:
        """Calculate optimal spread based on Avellaneda-Stoikov formula."""
        term1 = self.gamma * self.sigma ** 2 * time_remaining
        term2 = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        return term1 + term2

    def _cancel_all_live(self, book: OrderBook, actions: list) -> None:
        """Cancel every outstanding order this MM has in the book."""
        stale = []
        for oid, info in self.active_orders.items():
            info["age"] += 1
            if oid not in book.order_index:
                stale.append(oid)
            elif info["age"] >= self.ttl:
                actions.append(("cancel", oid))
                stale.append(oid)
                self.cancel += 1
            else:
                # FIX: also cancel non-stale orders so we can re-quote
                # at fresh prices every tick (two-sided obligation)
                actions.append(("cancel", oid))
                stale.append(oid)

        for oid in stale:
            self.active_orders.pop(oid, None)

        self.last_bid_id = None
        self.last_ask_id = None

    def act(self, t: int, book: OrderBook) -> Action:
        actions = []

        # ── step 1: cancel ALL existing quotes ──
        self._cancel_all_live(book, actions)

        # ── step 2: determine mid price ──
        mid_price = self._get_mid(book)
        bb, ba = book.best_bid(), book.best_ask()

        # ── step 3: Avellaneda-Stoikov reservation price ──
        time_remaining = max(0.1, 1.0 - t / self.T)
        raw_skew = self.inventory * self.gamma * self.sigma ** 2 * time_remaining

        # FIX: cap the skew so the MM never drifts too far from mid
        max_skew = self.max_skew_ticks * book.tick
        capped_skew = max(-max_skew, min(raw_skew, max_skew))

        reserve_price = mid_price - capped_skew
        optimal_spread = self.optimal_spread(time_remaining)

        bid_price = reserve_price - optimal_spread / 2
        ask_price = reserve_price + optimal_spread / 2

        # ── step 4: safety clamps ──
        # prevent extreme quotes: bound within a reasonable range around mid
        max_offset = self.max_skew_ticks * book.tick
        bid_price = max(bid_price, mid_price - max_offset)
        ask_price = min(ask_price, mid_price + max_offset)

        # prevent bid above best ask or ask below best bid
        if np.isfinite(ba) and bid_price >= ba:
            bid_price = ba - book.tick
        if np.isfinite(bb) and bb > 0 and ask_price <= bb:
            ask_price = bb + book.tick

        # prevent crossed quotes
        if bid_price >= ask_price:
            bid_price = mid_price - book.tick
            ask_price = mid_price + book.tick

        # ── step 5: inventory-aware sizing ──
        base_qty = 20
        if abs(self.inventory) > self.inventory_limit:
            # skew qty to pull inventory back toward zero
            # if short (inv < 0): increase bid qty, decrease ask qty
            # if long  (inv > 0): increase ask qty, decrease bid qty
            ratio = min(abs(self.inventory) / self.inventory_limit, 3.0)
            if self.inventory < 0:
                bid_qty = int(base_qty * ratio)
                ask_qty = max(5, int(base_qty / ratio))
            else:
                bid_qty = max(5, int(base_qty / ratio))
                ask_qty = int(base_qty * ratio)
        else:
            bid_qty = base_qty
            ask_qty = base_qty

        print(f"MarketMakerAS act: t={t}, inventory={self.inventory}, "
              f"mid={mid_price:.2f}, bid_px={bid_price:.2f}, ask_px={ask_price:.2f}, "
              f"bid_qty={bid_qty}, ask_qty={ask_qty}")

        # ── step 6: ALWAYS post both sides ──
        bid = Order(self.new_oid(), self.trader_id, "buy", bid_qty,
                    price=bid_price, ts=t)
        actions.append(bid)
        self.active_orders[bid.order_id] = {
            "side": "buy", "price": bid_price, "age": 0, "qty": bid_qty}
        self.last_bid_id = bid.order_id

        ask = Order(self.new_oid(), self.trader_id, "sell", ask_qty,
                    price=ask_price, ts=t)
        actions.append(ask)
        self.active_orders[ask.order_id] = {
            "side": "sell", "price": ask_price, "age": 0, "qty": ask_qty}
        self.last_ask_id = ask.order_id

        print(f'Cancellations:{self.cancel}')

        return actions if actions else None

class MarketMaker(BaseAgent):
    """
    Simple background market maker:
    - posts one limit order per action around mid
    - configurable spread range and order size
    """

    def __init__(
        self,
        trader_id: int,
        rng: np.random.Generator,
        max_spread_ticks: int = 10,
        min_qty: int = 1,
        max_qty: int = 10,
    ):
        super().__init__(trader_id, rng)
        self.max_spread_ticks = max_spread_ticks
        self.min_qty = min_qty
        self.max_qty = max_qty

    def act(self, t: int, book: OrderBook) -> Action:
        r = self.rng.random()

        side: Side = "buy" if r < 0.55 else "sell"
        qty = int(self.rng.integers(self.min_qty, self.max_qty + 1))

        mid = self._get_mid(book)

        tick = book.tick
        offset_ticks = int(self.rng.integers(0, self.max_spread_ticks + 1))

        if side == "buy":
            px = mid - offset_ticks * tick
        if side == "sell":
            px = mid + offset_ticks * tick

        return Order(self.new_oid(), self.trader_id, side, qty, price=float(px), ts=t)


class InstitutionalTrader(BaseAgent):
    def __init__(self, trader_id: int, rng: np.random.Generator,
                 participation_rate: float = 0.05,
                 use_iceberg_prob: float = 0.8,
                 dark_fraction: float = 0.05,
                 peak_range=(30, 50),
                 total_range=(100, 150),
                 price_mode: str = "join"):  # "join" or "improve"
        super().__init__(trader_id, rng)
        self.participation_rate = participation_rate
        self.use_iceberg_prob = use_iceberg_prob
        self.dark_fraction = dark_fraction
        self.peak_range = peak_range
        self.total_range = total_range
        self.price_mode = price_mode

        # # net signed inventory: positive = long, negative = short
        # # used to bias side selection so the trader mean-reverts toward zero
        # self.inventory: int = 0
        # # maximum inventory magnitude before side probability is fully skewed
        # self.inventory_limit: int = 500

    def act(self, t: int, book: OrderBook) -> Action:
        # 1) If currently executing iceberg, keep working it
        if self.iceberg is not None:
            return self.iceberg_step(t, book)

        # 2) Decide whether to initiate a parent order
        if self.rng.random() > self.participation_rate:
            return None

        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        total_qty = int(self.rng.integers(self.total_range[0], self.total_range[1] + 1))

        # If you don't have quotes, just skip
        bb, ba = book.best_bid(), book.best_ask()
        if bb <= 0 or not np.isfinite(ba):
            return None

        peak = int(self.rng.integers(self.peak_range[0], self.peak_range[1] + 1))

        # price selection (simple)
        if side == "buy":
            px = bb if self.price_mode == "join" else min(bb + book.tick, ba - book.tick)
        else:
            px = ba if self.price_mode == "join" else max(ba - book.tick, bb + book.tick)

        # 3) Choose iceberg vs one-shot market
        if self.rng.random() < self.use_iceberg_prob:
            self.iceberg_start(side=side, total_qty=total_qty, peak=peak, price=px)
            print('--------------------- ICEBERG PLACED --------------------------')
            return self.iceberg_step(t, book)  # post first slice
        else:
            return Order(self.new_oid(), self.trader_id, side, total_qty, price=None, ts=t)

    def act_dark(self, t: int, dark_pool) -> Optional[list]:
        """
        Submit a dark pool order if the participation check passes.

        Uses the same participation rate and order sizing as act(), but submits
        to the dark pool instead of the lit book. Returns the list of dark pool
        trades produced by the submission, or None if the agent chose not to act.

        :param t: current simulation timestep.
        :param dark_pool: DarkPool instance to submit to.
        :return: list of Trade objects from dark pool matching, or None.
        """

        if self.dark_fraction <= 0 or self.rng.random() > self.dark_fraction:
            return None

        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        qty = int(self.rng.integers(self.total_range[0], self.total_range[1] + 1))
        order = DarkPoolOrder(
            order_id=self.new_oid(),
            trader_id=self.trader_id,
            side=side,
            qty=qty,
            ts=t,
        )
        return dark_pool.submit_order(order)
from dataclasses import dataclass

class InformedTrader(BaseAgent):
    def __init__(
        self,
        trader_id: int,
        rng: np.random.Generator,
        fundamental: FundamentalProcess,
        sigma_s: float = 0.10,
        entry_threshold: float = 0.05,
        aggressive_threshold: float = 0.20,
        base_qty: int = 10,
        max_qty: int = 50,
        participation_rate: float = 0.8,
        dark_fraction: float = 0.20,   # <-- 20% of volume routed to dark
    ):
        super().__init__(trader_id, rng)
        self.fundamental = fundamental
        self.sigma_s = sigma_s
        self.entry_threshold = entry_threshold
        self.aggressive_threshold = aggressive_threshold
        self.base_qty = base_qty
        self.max_qty = max_qty
        self.participation_rate = participation_rate
        self.dark_fraction = dark_fraction

    def _size_order(self, edge: float) -> int:
        scale = min(abs(edge) / self.aggressive_threshold, 1.0)
        return max(1, int(self.base_qty + scale * (self.max_qty - self.base_qty)))

    def act(self, t: int, book: OrderBook) -> Action:
        if self.rng.random() > self.participation_rate:
            return None

        signal = self.fundamental.observe(self.sigma_s, self.rng)

        mid = self._get_mid(book)

        edge = signal - mid
        if abs(edge) < self.entry_threshold:
            return None

        side: Side = "buy" if edge > 0 else "sell"
        total_qty = self._size_order(edge)

        # -------- split lit / dark --------
        dark_qty = int(total_qty * self.dark_fraction)
        lit_qty = total_qty - dark_qty

        # store dark order info so simulation can route it
        self.pending_dark_order = (side, dark_qty) if dark_qty > 0 else None

        # -------- lit order --------
        if abs(edge) >= self.aggressive_threshold:
            return Order(self.new_oid(), self.trader_id, side, lit_qty, price=None, ts=t)

        bb, ba = book.best_bid(), book.best_ask()
        px = ba if side == "buy" else bb

        if not np.isfinite(px):
            return Order(self.new_oid(), self.trader_id, side, lit_qty, price=None, ts=t)

        return Order(self.new_oid(), self.trader_id, side, lit_qty, price=float(px), ts=t)

    def act_dark(self, t: int, dark_pool):
        """
        Send the stored dark portion of the informed order.
        """
        if not hasattr(self, "pending_dark_order") or self.pending_dark_order is None:
            return None

        side, qty = self.pending_dark_order
        self.pending_dark_order = None

        order = DarkPoolOrder(
            order_id=self.new_oid(),
            trader_id=self.trader_id,
            side=side,
            qty=qty,
            ts=t,
        )

        return dark_pool.submit_order(order)