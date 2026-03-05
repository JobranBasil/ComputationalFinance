from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
import numpy as np
import math

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

@dataclass
class BaseAgent:
    trader_id: int
    rng: np.random.Generator
    _next_order_id: int = 1

    # NEW: optional iceberg currently being executed by this agent
    iceberg: Optional[IcebergOrder] = None

    def new_oid(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        # make ids globally unique-ish by namespacing with trader_id
        return int(self.trader_id * 1_000_000 + oid)

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
        participation_rate: float = 0.3,
        market_prob: float = 0.5,
        sign_persistence: float = 0.7,
        max_depth_ticks: int = 3,
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

        qty = int(self.rng.integers(1, 5))

        # market order
        if self.rng.random() < self.market_prob:
            return Order(self.new_oid(), self.trader_id, side, qty, price=None, ts=t)

        # limit order around mid
        mid = book.mid_price()
        if not np.isfinite(mid):
            mid = 100.0

        tick = book.tick
        offset_ticks = int(self.rng.integers(0, self.max_depth_ticks + 1))

        if side == "buy":
            px = mid - offset_ticks * tick
        else:
            px = mid + offset_ticks * tick

        return Order(self.new_oid(), self.trader_id, side, qty, price=float(px), ts=t)

class MarketMakerAS(BaseAgent):
    """
    Market maker with inventory risk adjusted spread (Avellaneda-Stoikov)
    """
    def __init__(self, 
                 trader_id: int,
                 rng: np.random.Generator,
                 kappa: float, # Order‐book liquidity parameter (κ)
                 gamma: float, # Inventory risk aversion (γ)
                 sigma: float, # Market volatility (σ)
                 horizon: float, # Time horizon
                 A: float, # Baseline arrival rate
                 mid_price_threshold: float = 0.50,
                 inventory_threshold: int = 2,
                 ):
        super().__init__(trader_id, rng)
        self.kappa = kappa
        self.gamma = gamma
        self.sigma = sigma
        self.T = horizon
        self.inventory = 0
        self.A = A
        #If we want to track PnL, we can add a cash attribute and update it on trades
        self.cash = 0.0
        self.order_index: Dict[int, Tuple[Side, float]] = {}  # order_id -> (side, price)

        # Thresholds
        self.mid_price_threshold = mid_price_threshold
        self.inventory_threshold = inventory_threshold
    
    def update_inventory(self, trade: Trade):
        """Update inventory based on executed trade."""
        if trade.aggressor_side == "buy":
            self.inventory -= trade.qty  # sold to buyer, decrease inventory
        else:
            self.inventory += trade.qty  # bought from seller, increase inventory
    
    def optimal_spread(self, time_remaining: float) -> float:
        """Calculate optimal spread based on Avellaneda-Stoikov formula."""
        term1 = self.gamma * self.sigma**2 * time_remaining
        term2 = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        return term1 + term2
    
    def arrival_intensity(self, delta: float) -> float:
        """Exponential arrival intensity for orders at distance δ."""
        return self.A * math.exp(-self.kappa * delta)
    
    def act(self, t: int, book: OrderBook) -> Action:
        # Calculate mid-price and optimal spread
        bb, ba = book.best_bid(), book.best_ask()
        if bb <= 0 or not np.isfinite(ba):
            mid_price = 100.0  # default mid if no quotes
        else:
            mid_price = (bb + ba) / 2
        
        #Calculate reservation price and optimal quotes
        time_remaining = self.T - t
        rerserve_price = mid_price - self.inventory * self.gamma * self.sigma**2 * time_remaining
        optimal_spread = self.optimal_spread(time_remaining)

        bid_price = rerserve_price - optimal_spread / 2
        ask_price = rerserve_price + optimal_spread / 2


        #Quote orders according to arrival_intensity or probability of getting filled
        delta_bid = mid_price - bid_price
        delta_ask = ask_price - mid_price

        #Fill probability based on arrival intensity (dt = 1 for simplicity)
        fill_prob_bid = self.arrival_intensity(delta_bid)
        fill_prob_ask = self.arrival_intensity(delta_ask)

        #Randomly decide to place/cancel orders based on fill probabilities
        r = self.rng.random()
        if r < fill_prob_bid: 
            bid = Order(self.new_oid(), self.trader_id, "buy", qty=1, price=bid_price, ts=t)
            self.update_inventory(bid)
            return bid   
        elif r < fill_prob_bid + fill_prob_ask: 
            ask = Order(self.new_oid(), self.trader_id, "sell", qty=1, price=ask_price, ts=t)
            self.update_inventory(ask)
            return ask
        else:  
            return None


class MarketMaker(BaseAgent):
    """
    Minimal placeholder:
    - posts one bid and one ask at the current bests (degenerate, but exercises add_limit)
    - occasionally cancels a random existing order from the book
    """

    def act(self, t: int, book: OrderBook) -> Action:
        #if self.rng.random() > 0.9:
            #return None
        
        r = self.rng.random()

        side: Side = "buy" if r < 0.55 else "sell"
        qty = int(self.rng.integers(1, 10))
        bb, ba = book.best_bid(), book.best_ask()

        # limit order around mid
        mid = book.mid_price()
        if not np.isfinite(mid):
            mid = 100.0

        tick = book.tick
        offset_ticks = int(self.rng.integers(0, 10))

        if side == "buy":
            px = mid - offset_ticks * tick
        if side == "sell":
            px = mid + offset_ticks * tick

        return Order(self.new_oid(), self.trader_id, side, qty, price=float(px), ts=t)


class InstitutionalTrader(BaseAgent):
    """
    Minimal placeholder:
    - rarely acts
    - when acts, sends a larger market order (exercises matching)
    """

    def act(self, t: int, book: OrderBook) -> Action:
        if self.rng.random() > 0.05:
            return None

        side: Side = "buy" if self.rng.random() < 0.5 else "sell"
        qty = int(self.rng.integers(10, 50))
        return Order(self.new_oid(), self.trader_id, side, qty, price=None, ts=t)