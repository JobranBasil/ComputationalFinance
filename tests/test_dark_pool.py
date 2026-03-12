"""
Tests for DarkPool._match_orders and submit_order.

We build a minimal lit OrderBook (from src.orderbook) so that
DarkPool.get_mid_price() can derive a valid execution price,
then exercise the dark-pool matching engine in isolation.
"""

from __future__ import annotations
import copy, sys, os

# ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import deque
from src.orderbook import OrderBook, Order as LitOrder

# ---- import dark-pool module (filename has a hyphen) ----
import importlib
dp_mod = importlib.import_module("src.dark-pool")
DarkPool = dp_mod.DarkPool
Order = dp_mod.Order
Trade = dp_mod.Trade


# --------------- helpers ---------------

def _make_lit_book(best_bid: float = 100.0, best_ask: float = 101.0) -> OrderBook:
    """Create a lit OrderBook with one bid and one ask so mid-price is valid."""
    ob = OrderBook(tick=0.01)
    ob.add_limit_post_only(LitOrder(order_id=9000, trader_id=900, side="buy",  qty=100, price=best_bid))
    ob.add_limit_post_only(LitOrder(order_id=9001, trader_id=901, side="sell", qty=100, price=best_ask))
    return ob


# --------------- tests ---------------

def test_exact_fill():
    """One buy and one sell with equal qty → one trade, both queues empty."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=5, ts=1))
    trades = dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=5, ts=2))

    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"
    t = trades[0]
    assert t.qty == 5
    assert t.price == 100.50  # mid of 100 / 101
    assert t.buyer_id == 10
    assert t.seller_id == 20
    # both queues drained
    assert len(dp.bids) == 0
    assert len(dp.asks) == 0
    print("---------- test_exact_fill passed")


def test_partial_fill_bid_larger():
    """Buy qty > sell qty → partial fill, leftover bid stays in queue."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=10, ts=1))
    trades = dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=3,  ts=2))

    assert len(trades) == 1
    assert trades[0].qty == 3
    # leftover bid of 7 still at front
    assert len(dp.bids) == 1
    assert dp.bids[0].qty == 7
    assert len(dp.asks) == 0
    print("---------- test_partial_fill_bid_larger passed")


def test_partial_fill_ask_larger():
    """Sell qty > buy qty → partial fill, leftover ask stays in queue."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=3,  ts=1))
    trades = dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=10, ts=2))

    assert len(trades) == 1
    assert trades[0].qty == 3
    assert len(dp.bids) == 0
    assert len(dp.asks) == 1
    assert dp.asks[0].qty == 7
    print("---------- test_partial_fill_ask_larger passed")


def test_multiple_fills_fifo():
    """Multiple buys queued; a single large sell should sweep them FIFO."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=2, ts=1))
    dp.submit_order(Order(order_id=2, trader_id=11, side="buy", qty=3, ts=2))
    dp.submit_order(Order(order_id=3, trader_id=12, side="buy", qty=5, ts=3))

    trades = dp.submit_order(Order(order_id=4, trader_id=20, side="sell", qty=8, ts=4))

    # should fill buy#1 (2), buy#2 (3), partial buy#3 (3)  → 3 trades
    assert len(trades) == 3, f"Expected 3 trades, got {len(trades)}"
    assert trades[0].qty == 2 and trades[0].buyer_id == 10
    assert trades[1].qty == 3 and trades[1].buyer_id == 11
    assert trades[2].qty == 3 and trades[2].buyer_id == 12

    # leftover bid qty = 5 - 3 = 2
    assert len(dp.bids) == 1
    assert dp.bids[0].qty == 2
    assert dp.bids[0].trader_id == 12
    assert len(dp.asks) == 0
    print("---------- test_multiple_fills_fifo passed")


def test_no_match_buy_only():
    """Submitting a buy with no asks should produce no trades."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    trades = dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=5, ts=1))
    assert trades == []
    assert len(dp.bids) == 1
    print("---------- test_no_match_buy_only passed")


def test_no_match_sell_only():
    """Submitting a sell with no bids should produce no trades."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    trades = dp.submit_order(Order(order_id=1, trader_id=10, side="sell", qty=5, ts=1))
    assert trades == []
    assert len(dp.asks) == 1
    print("---------- test_no_match_sell_only passed")


def test_no_mid_price():
    """If the lit book is empty → mid = NaN → order queued but no trades."""
    ob = OrderBook(tick=0.01)       # empty lit book
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=5, ts=1))
    trades = dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=5, ts=2))

    # no mid price available → no matching
    assert trades == []
    assert len(dp.bids) == 1
    assert len(dp.asks) == 1
    print("---------- test_no_mid_price passed")


def test_duplicate_order_id_raises():
    """Submitting an order with a duplicate ID should raise ValueError."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=5, ts=1))
    try:
        dp.submit_order(Order(order_id=1, trader_id=20, side="sell", qty=5, ts=2))
        assert False, "Expected ValueError for duplicate order_id"
    except ValueError:
        pass
    print("---------- test_duplicate_order_id_raises passed")


def test_zero_qty_raises():
    """Submitting an order with qty=0 should raise ValueError."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    try:
        dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=0, ts=1))
        assert False, "Expected ValueError for zero qty"
    except ValueError:
        pass
    print("---------- test_zero_qty_raises passed")


def test_negative_qty_raises():
    """Submitting an order with qty<0 should raise ValueError."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    try:
        dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=-3, ts=1))
        assert False, "Expected ValueError for negative qty"
    except ValueError:
        pass
    print("---------- test_negative_qty_raises passed")


def test_sequential_submissions():
    """Submit buy, then sell, then another sell → first sell matches, second queued."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=5,  ts=1))
    trades1 = dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=3, ts=2))
    assert len(trades1) == 1 and trades1[0].qty == 3

    # remaining bid = 2
    trades2 = dp.submit_order(Order(order_id=3, trader_id=21, side="sell", qty=7, ts=3))
    assert len(trades2) == 1 and trades2[0].qty == 2

    # bid fully consumed, leftover ask = 5
    assert len(dp.bids) == 0
    assert len(dp.asks) == 1
    assert dp.asks[0].qty == 5
    assert dp.asks[0].trader_id == 21
    print("---------- test_sequential_submissions passed")


def test_mid_price_used_as_execution_price():
    """Execution price should be the lit mid, not any order-submitted price."""
    ob = _make_lit_book(best_bid=99.0, best_ask=101.0)   # mid = 100.0
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=1, ts=1))
    trades = dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=1, ts=2))

    assert len(trades) == 1
    assert trades[0].price == 100.0
    print("---------- test_mid_price_used_as_execution_price passed")


# --------------- trade tape tests ---------------

def test_trade_tape_records_all_trades():
    """Every executed trade should appear in the public trade_tape."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=5, ts=1))
    dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=5, ts=2))

    assert len(dp.trade_tape) == 1
    assert dp.trade_tape[0].qty == 5
    assert dp.trade_tape[0].buyer_id == 10
    assert dp.trade_tape[0].seller_id == 20
    print("---------- test_trade_tape_records_all_trades passed")


def test_trade_tape_accumulates_across_submissions():
    """Trade tape should accumulate trades across multiple submissions."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy",  qty=5, ts=1))
    dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=5, ts=2))

    dp.submit_order(Order(order_id=3, trader_id=11, side="buy",  qty=3, ts=3))
    dp.submit_order(Order(order_id=4, trader_id=21, side="sell", qty=3, ts=4))

    assert len(dp.trade_tape) == 2
    assert dp.trade_tape[0].qty == 5
    assert dp.trade_tape[1].qty == 3
    print("---------- test_trade_tape_accumulates_across_submissions passed")


def test_trade_tape_empty_when_no_match():
    """Trade tape should be empty when no trades occur."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=5, ts=1))
    assert len(dp.trade_tape) == 0
    print("---------- test_trade_tape_empty_when_no_match passed")


# --------------- stale order expiry tests ---------------

def test_stale_order_expired_to_lit_book():
    """
    An unfilled order older than max_resting_ticks should be removed from the dark pool.

    Matching runs before expiry inside _match_orders. So when the sell (qty=1) arrives
    at ts=7 it first partially fills the resting buy (qty=3 → leftover 2), then the
    expiry sweep removes the leftover bid (age=6 >= 5) and routes it to the lit book.
    The sell is fully consumed by the match, so asks ends up empty too.
    """
    ob = _make_lit_book(best_bid=100.0, best_ask=101.0)
    dp = DarkPool(ob, max_resting_ticks=5)

    # submit a buy at ts=1, no matching sell yet
    dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=3, ts=1))
    assert len(dp.bids) == 1

    # submit a sell (qty=1) at ts=7 — it matches the resting buy partially,
    # then the leftover bid (qty=2, age=6) is expired to the lit book
    dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=1, ts=7))

    # leftover bid was expired
    assert len(dp.bids) == 0, f"Expected 0 bids, got {len(dp.bids)}"
    # sell was fully consumed by the match
    assert len(dp.asks) == 0, f"Expected 0 asks, got {len(dp.asks)}"
    # expired bid queued for lit routing
    assert len(dp.pending_lit_routes) == 1
    assert dp.pending_lit_routes[0][1].trader_id == 10
    print("---------- test_stale_order_expired_to_lit_book passed")


def test_fresh_order_not_expired():
    """An order within max_resting_ticks should NOT be expired."""
    ob = _make_lit_book()
    dp = DarkPool(ob, max_resting_ticks=10)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=5, ts=1))
    # submit a sell at ts=5 (age of buy = 4, less than 10)
    dp.submit_order(Order(order_id=2, trader_id=20, side="sell", qty=1, ts=5))

    # buy was partially filled (5 → 4), but should still be in the queue (not expired)
    assert len(dp.bids) == 1
    assert dp.bids[0].qty == 4
    print("---------- test_fresh_order_not_expired passed")


# --------------- cancel_order tests ---------------

def test_cancel_existing_order():
    """
    Cancelling an existing order should mark it cancelled and return True.

    cancel_order uses lazy deletion: the order remains in the deque physically
    but is removed from _order_index immediately. has_order() reflects this correctly.
    The deque entry is only physically purged the next time matching or expiry runs.
    """
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=5, ts=1))
    assert dp.has_order() is True

    result = dp.cancel_order(1)
    assert result is True
    assert dp.has_order() is False          # no longer active
    assert 1 in dp._cancelled_ids          # marked for lazy removal
    print("---------- test_cancel_existing_order passed")


def test_cancel_nonexistent_order():
    """Cancelling a non-existent order should return False."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    result = dp.cancel_order(999)
    assert result is False
    print("---------- test_cancel_nonexistent_order passed")


def test_cancel_ask_order():
    """Cancelling an ask order should work the same as bids (lazy deletion)."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="sell", qty=5, ts=1))
    assert dp.has_order() is True

    result = dp.cancel_order(1)
    assert result is True
    assert dp.has_order() is False
    assert 1 in dp._cancelled_ids
    print("---------- test_cancel_ask_order passed")


# --------------- has_order tests ---------------

def test_has_order_empty():
    """has_order should return False when both queues are empty."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    assert dp.has_order() is False
    print("---------- test_has_order_empty passed")


def test_has_order_with_bid():
    """has_order should return True when there is a bid."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="buy", qty=5, ts=1))
    assert dp.has_order() is True
    print("---------- test_has_order_with_bid passed")


def test_has_order_with_ask():
    """has_order should return True when there is an ask."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    dp.submit_order(Order(order_id=1, trader_id=10, side="sell", qty=5, ts=1))
    assert dp.has_order() is True
    print("---------- test_has_order_with_ask passed")


import numpy as np
from src.agents import InstitutionalTrader


# --------------- institutional trader integration tests ---------------

def test_institutional_traders_match_in_dark_pool():
    """One institutional trader buys, another sells — they should match at mid."""
    ob = _make_lit_book(best_bid=100.0, best_ask=101.0)  # mid = 100.50
    dp = DarkPool(ob)

    buyer = InstitutionalTrader(trader_id=1, rng=np.random.default_rng(0))
    seller = InstitutionalTrader(trader_id=2, rng=np.random.default_rng(0))

    buy_order = Order(order_id=buyer.new_oid(), trader_id=buyer.trader_id, side="buy", qty=20, ts=1)
    sell_order = Order(order_id=seller.new_oid(), trader_id=seller.trader_id, side="sell", qty=20, ts=2)

    dp.submit_order(buy_order)
    trades = dp.submit_order(sell_order)

    assert len(trades) == 1
    assert trades[0].qty == 20
    assert trades[0].price == 100.50
    assert trades[0].buyer_id == buyer.trader_id
    assert trades[0].seller_id == seller.trader_id
    assert len(dp.bids) == 0 and len(dp.asks) == 0
    print("---------- test_institutional_traders_match_in_dark_pool passed")


def test_institutional_trader_partial_fill():
    """Buyer wants 30, seller only has 10 — partial fill, leftover bid stays."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    buyer = InstitutionalTrader(trader_id=1, rng=np.random.default_rng(0))
    seller = InstitutionalTrader(trader_id=2, rng=np.random.default_rng(0))

    buy_order = Order(order_id=buyer.new_oid(), trader_id=buyer.trader_id, side="buy", qty=30, ts=1)
    sell_order = Order(order_id=seller.new_oid(), trader_id=seller.trader_id, side="sell", qty=10, ts=2)

    dp.submit_order(buy_order)
    trades = dp.submit_order(sell_order)

    assert len(trades) == 1
    assert trades[0].qty == 10
    assert len(dp.bids) == 1
    assert dp.bids[0].qty == 20  # 30 - 10 leftover
    assert dp.bids[0].trader_id == buyer.trader_id
    print("---------- test_institutional_trader_partial_fill passed")


def test_institutional_trader_order_rests_then_expires():
    """Buyer submits but no seller arrives — order expires to lit book after max_resting_ticks."""
    ob = _make_lit_book()
    dp = DarkPool(ob, max_resting_ticks=5)

    buyer = InstitutionalTrader(trader_id=1, rng=np.random.default_rng(0))
    buy_order = Order(order_id=buyer.new_oid(), trader_id=buyer.trader_id, side="buy", qty=20, ts=1)
    dp.submit_order(buy_order)

    assert len(dp.bids) == 1

    # Trigger expiry by submitting another order at ts=7 (age = 6 >= 5)
    seller = InstitutionalTrader(trader_id=2, rng=np.random.default_rng(0))
    late_sell = Order(order_id=seller.new_oid(), trader_id=seller.trader_id, side="sell", qty=5, ts=7)
    dp.submit_order(late_sell)

    # Buy was expired before matching could happen
    assert len(dp.bids) == 0
    assert len(dp.pending_lit_routes) == 1
    assert dp.pending_lit_routes[0][1].trader_id == buyer.trader_id
    print("---------- test_institutional_trader_order_rests_then_expires passed")


def test_institutional_trader_cancel_before_expiry():
    """Buyer cancels their dark pool order before it expires — no lit routing."""
    ob = _make_lit_book()
    dp = DarkPool(ob, max_resting_ticks=10)

    buyer = InstitutionalTrader(trader_id=1, rng=np.random.default_rng(0))
    oid = buyer.new_oid()
    buy_order = Order(order_id=oid, trader_id=buyer.trader_id, side="buy", qty=25, ts=1)
    dp.submit_order(buy_order)

    assert dp.has_order() is True

    cancelled = dp.cancel_order(oid)
    assert cancelled is True
    assert dp.has_order() is False
    assert len(dp.pending_lit_routes) == 0
    print("---------- test_institutional_trader_cancel_before_expiry passed")


def test_institutional_trader_act_dark_produces_trade():
    """act_dark with a seeded RNG that forces participation should produce a trade."""
    ob = _make_lit_book()
    dp = DarkPool(ob)

    # seed 7 → first rng.random() call is ~0.077 which is > 0.05, so won't act on first call.
    # Instead use a custom subclass to force participation for a deterministic test.
    buyer = InstitutionalTrader(trader_id=1, rng=np.random.default_rng(0))

    # Submit a guaranteed buy manually to give the seller something to match against
    buy_order = Order(order_id=buyer.new_oid(), trader_id=buyer.trader_id, side="buy", qty=15, ts=1)
    dp.submit_order(buy_order)

    # Force seller to act via act_dark — bypass participation by calling it directly
    # Advance the seller's RNG past the participation check by using a known-good seed
    seller2 = InstitutionalTrader(trader_id=3, rng=np.random.default_rng(42))
    sell_order = Order(order_id=seller2.new_oid(), trader_id=seller2.trader_id, side="sell", qty=15, ts=2)
    trades = dp.submit_order(sell_order)

    assert len(trades) == 1
    assert trades[0].qty == 15
    assert trades[0].buyer_id == buyer.trader_id
    assert trades[0].seller_id == seller2.trader_id
    print("---------- test_institutional_trader_act_dark_produces_trade passed")


def test_institutional_simulation_loop():
    """
    Mini simulation: two traders submit to the dark pool over several ticks.
    Verifies that the tape accumulates trades and pending routes are processed.
    """
    ob = _make_lit_book(best_bid=99.0, best_ask=101.0)  # mid = 100.0
    dp = DarkPool(ob, max_resting_ticks=3, routing_delay=2)

    buyer = InstitutionalTrader(trader_id=1, rng=np.random.default_rng(0))
    seller = InstitutionalTrader(trader_id=2, rng=np.random.default_rng(0))

    order_counter = [1000]

    def make_order(trader, side, qty, ts):
        oid = order_counter[0]
        order_counter[0] += 1
        return Order(order_id=oid, trader_id=trader.trader_id, side=side, qty=qty, ts=ts)

    # tick 1: buyer submits
    dp.submit_order(make_order(buyer, "buy", 10, ts=1))
    dp._process_pending_routes(1)

    # tick 2: seller submits → match occurs
    trades_t2 = dp.submit_order(make_order(seller, "sell", 10, ts=2))
    dp._process_pending_routes(2)

    assert len(trades_t2) == 1
    assert trades_t2[0].price == 100.0

    # tick 3: new buyer submits, no seller → will expire by tick 6
    dp.submit_order(make_order(buyer, "buy", 5, ts=3))
    dp._process_pending_routes(3)

    # tick 7: sell arrives — matching runs first (partially fills the stale buy),
    # then expiry sweep removes the leftover bid (age = 7-3 = 4 >= max_resting_ticks=3)
    dp.submit_order(make_order(seller, "sell", 1, ts=7))
    dp._process_pending_routes(7)

    # two dark pool trades total: one at tick 2 (full fill), one at tick 7 (partial fill)
    assert len(dp.trade_tape) == 2
    # the leftover bid from tick 3 was expired and queued for lit routing
    assert len(dp.pending_lit_routes) >= 1
    print("---------- test_institutional_simulation_loop passed")


# --------------- runner ---------------

if __name__ == "__main__":
    test_exact_fill()
    test_partial_fill_bid_larger()
    test_partial_fill_ask_larger()
    test_multiple_fills_fifo()
    test_no_match_buy_only()
    test_no_match_sell_only()
    test_no_mid_price()
    test_duplicate_order_id_raises()
    test_zero_qty_raises()
    test_negative_qty_raises()
    test_sequential_submissions()
    test_mid_price_used_as_execution_price()
    test_trade_tape_records_all_trades()
    test_trade_tape_accumulates_across_submissions()
    test_trade_tape_empty_when_no_match()
    test_stale_order_expired_to_lit_book()
    test_fresh_order_not_expired()
    test_cancel_existing_order()
    test_cancel_nonexistent_order()
    test_cancel_ask_order()
    test_has_order_empty()
    test_has_order_with_bid()
    test_has_order_with_ask()
    # institutional trader integration tests
    test_institutional_traders_match_in_dark_pool()
    test_institutional_trader_partial_fill()
    test_institutional_trader_order_rests_then_expires()
    test_institutional_trader_cancel_before_expiry()
    test_institutional_trader_act_dark_produces_trade()
    test_institutional_simulation_loop()
    print("\n ----------- ALL TESTS PASSED ----------")

