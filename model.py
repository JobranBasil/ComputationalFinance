# mesa model for ABM orderbook sim

from __future__ import annotations

import importlib
import math
import numpy as np
import mesa

from src.fundemental import FundamentalProcess
from src.orderbook import OrderBook, Order
from src.agents import (
    NoiseTrader,
    MarketMaker,
    InstitutionalTrader,
    MarketMakerAS,
    InformedTrader,
    Action,
)
from src.dark_pool import DarkPool, Order as DarkOrder


# ── helpers (copied from run.py) ──────────────────────────────────────

def apply_action(book: OrderBook, action) -> list:
    if action is None:
        return []
    if isinstance(action, list):
        out = []
        for a in action:
            out.extend(apply_action(book, a))
        return out
    if isinstance(action, tuple) and action[0] == "cancel":
        book.cancel(action[1])
        return []
    if isinstance(action, Order):
        return book.execute_market(action) if action.price is None else book.add_limit(action)
    raise TypeError(f"Unknown action: {type(action)}")


def seed_initial_book(book, best_bid, best_ask, levels, rng):
    for i in range(levels):
        book.add_limit_post_only(Order(
            order_id=10_000 + i, trader_id=-1, side="buy",
            qty=int(rng.integers(1, 10)),
            price=best_bid - book.tick * i, ts=0,
        ))
        book.add_limit_post_only(Order(
            order_id=20_000 + i, trader_id=-1, side="sell",
            qty=int(rng.integers(1, 10)),
            price=best_ask + book.tick * i, ts=0,
        ))


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
    bid = sum(q for _, q in book.top_n_levels("buy", n))
    ask = sum(q for _, q in book.top_n_levels("sell", n))
    return bid, ask


def microprice(book):
    bb, ba = book.best_bid(), book.best_ask()
    if not np.isfinite(bb) or not np.isfinite(ba):
        return np.nan
    qb = sum(o.qty for o in book.bids.get(bb, []))
    qa = sum(o.qty for o in book.asks.get(ba, []))
    d = qb + qa
    return (ba * qb + bb * qa) / d if d > 0 else np.nan


# ── Mesa wrapper agents ───────────────────────────────────────────────

class MesaAgentWrapper(mesa.Agent):
    # mesa wrapper

    def __init__(self, model, inner_agent):
        super().__init__(model)
        self.inner = inner_agent

    def step(self):
        # stepping is driven by the model, not the scheduler
        pass


# ── Mesa Model ────────────────────────────────────────────────────────

class MarketModel(mesa.Model):
    # params exposed as solara sliders

    def __init__(
        self,
        steps: int = 500,
        sigma_v: float = 0.03,
        start_price: float = 100.0,
        noise_count: int = 3,
        noise_participation: float = 0.90,
        noise_market_prob: float = 0.50,
        institutional_count: int = 2,
        iceberg_prob: float = 0.80,
        institutional_participation: float = 0.05,
        inst_dark_fraction: float = 0.05,
        informed_count: int = 1,
        mmas_gamma: float = 0.10,
        mmas_kappa: float = 50.0,
        mmas_sigma: float = 0.05,
        mmas_horizon: float = 5000.0,
        informed_participation: float = 0.50,
        informed_sigma_s: float = 0.08,
        dark_fraction: float = 0.20,
        seed: int = 42,
    ):
        super().__init__()
        self.num_steps = steps

        # derive sub-seeds
        max_agents = noise_count + institutional_count + informed_count + 10
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(max_agents + 10)
        _si = 0

        def _next_seed():
            nonlocal _si
            s = child_seeds[_si]
            _si += 1
            return s

        self._rng = np.random.default_rng(_next_seed())

        self.fundamental = FundamentalProcess(
            start=start_price + 0.075,
            sigma_v=sigma_v,
            rng=np.random.default_rng(_next_seed()),
        )

        self.book = OrderBook(tick=0.01, max_depth_levels=20)
        seed_initial_book(
            self.book,
            best_bid=start_price + 0.05,
            best_ask=start_price + 0.10,
            levels=10,
            rng=np.random.default_rng(_next_seed()),
        )

        self.dark_pool = DarkPool(
            lit_orderbook=self.book,
            max_resting_ticks=50,
            routing_delay=5,
            tape_delay=5,
        )

        # create agents
        self._inner_agents = []

        for i in range(noise_count):
            self._inner_agents.append(
                NoiseTrader(
                    trader_id=100 + i,
                    rng=np.random.default_rng(_next_seed()),
                    participation_rate=noise_participation,
                    market_prob=noise_market_prob,
                )
            )

        self._inner_agents.append(
            MarketMaker(
                trader_id=10,
                rng=np.random.default_rng(_next_seed()),
            )
        )

        self.mmas = MarketMakerAS(
            trader_id=20,
            rng=np.random.default_rng(_next_seed()),
            horizon=mmas_horizon,
            kappa=mmas_kappa,
            gamma=mmas_gamma,
            sigma=mmas_sigma,
        )
        self._inner_agents.append(self.mmas)

        for i in range(informed_count):
            inf = InformedTrader(
                trader_id=30 + i,
                rng=np.random.default_rng(_next_seed()),
                fundamental=self.fundamental,
                sigma_s=informed_sigma_s,
                participation_rate=informed_participation,
                dark_fraction=dark_fraction,
            )
            self._inner_agents.append(inf)
        # keep ref to first informed
        self.informed = next(
            (a for a in self._inner_agents if isinstance(a, InformedTrader)), None
        )

        for i in range(institutional_count):
            self._inner_agents.append(
                InstitutionalTrader(
                    trader_id=40 + i,
                    rng=np.random.default_rng(_next_seed()),
                    use_iceberg_prob=iceberg_prob,
                    participation_rate=institutional_participation,
                    dark_fraction=inst_dark_fraction,
                )
            )

        # register mesa wrappers
        initial_mid = (start_price + 0.05 + start_price + 0.10) / 2
        for ia in self._inner_agents:
            ia._last_valid_mid = initial_mid
            MesaAgentWrapper(self, ia)

        # time-series storage
        self.history: list[dict] = []
        self.snapshots: list[dict] = []
        self.latest_snapshot: dict | None = None
        self._t = 0
        self._last_mid = initial_mid      # gapless chart tracker
        self._last_spread = 0.05

    # ────────────────────────────────────────────────────────────────────

    def step(self):
        t = self._t

        # 1) advance fundamental
        self.fundamental.step()

        # 2) agents act
        trades_this_step = []
        for agent in self._inner_agents:
            if isinstance(agent, MarketMaker):
                n = int(self._rng.integers(1, 4))
            else:
                n = 1

            for _ in range(n):
                action = agent.act(t, self.book)
                trades = apply_action(self.book, action)
                trades_this_step.extend(trades)

            # dark pool submissions
            if isinstance(agent, (InstitutionalTrader, InformedTrader)):
                dp_trades = agent.act_dark(t, self.dark_pool)
                if dp_trades:
                    trades_this_step.extend(
                        [type("T", (), {"qty": tr.qty, "price": tr.price})() for tr in dp_trades]
                    )

        # 3) update MMAS inventory
        for tr in trades_this_step:
            if hasattr(tr, "maker_trader_id"):
                if tr.maker_trader_id == self.mmas.trader_id:
                    self.mmas.update_inventory(tr)
                    if tr.maker_order_id == self.mmas.last_bid_id:
                        self.mmas.last_bid_id = None
                    elif tr.maker_order_id == self.mmas.last_ask_id:
                        self.mmas.last_ask_id = None
                elif tr.taker_trader_id == self.mmas.trader_id:
                    self.mmas.update_inventory(tr)
                    if tr.taker_order_id == self.mmas.last_bid_id:
                        self.mmas.last_bid_id = None
                    elif tr.taker_order_id == self.mmas.last_ask_id:
                        self.mmas.last_ask_id = None

        # 4) dark pool tick
        self.dark_pool.tick(t)

        # 5) collect metrics
        mid = self.book.mid_price()
        sp = self.book.spread()
        bb = self.book.best_bid()
        ba = self.book.best_ask()
        bid_d, ask_d = total_visible_depth(self.book)
        dp_bid, dp_ask = self.dark_pool.queue_depth()
        dp_vol = self.dark_pool.recent_volume(current_ts=t, lookback=20)
        mp = microprice(self.book)

        vol = sum(tr.qty for tr in trades_this_step if hasattr(tr, "qty"))

        # use last valid value
        if np.isfinite(mid):
            self._last_mid = mid
        else:
            mid = self._last_mid

        if np.isfinite(sp) and sp >= 0:
            self._last_spread = sp
        else:
            sp = self._last_spread

        mispricing = abs(mid - self.fundamental.value)

        self.history.append({
            "t": t,
            "mid": mid,
            "fundamental": self.fundamental.value,
            "spread": sp,
            "best_bid": bb if np.isfinite(bb) else mid - sp / 2,
            "best_ask": ba if np.isfinite(ba) else mid + sp / 2,
            "obi": self.book.book_imbalance() if (bid_d + ask_d) > 0 else 0,
            "microprice": mp if np.isfinite(mp) else mid,
            "mispricing": mispricing,
            "volume": vol,
            "mmas_inventory": self.mmas.inventory,
            "bid_depth": bid_d,
            "ask_depth": ask_d,
            "dp_bid_depth": dp_bid,
            "dp_ask_depth": dp_ask,
            "dp_volume": dp_vol,
            "num_bid_levels": len(self.book.bid_prices),
            "num_ask_levels": len(self.book.ask_prices),
        })

        # latest snapshot for live view
        self.latest_snapshot = {
            "t": t,
            "bids": self.book.top_n_levels("buy", 10),
            "asks": self.book.top_n_levels("sell", 10),
        }

        # periodic snapshots
        snap_interval = max(1, self.num_steps // 20)
        if t % snap_interval == 0 or t == self.num_steps - 1:
            self.snapshots.append({
                "t": t,
                "bids": self.book.top_n_levels("buy", 10),
                "asks": self.book.top_n_levels("sell", 10),
            })

        self._t += 1

    def run_all(self):
        # run full sim
        for _ in range(self.num_steps):
            self.step()
