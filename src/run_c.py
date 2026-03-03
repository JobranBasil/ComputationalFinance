from .OrderBook_c import Order, OrderBook
from numpy.random import poisson

def simple_test1():
    ob = OrderBook()
    # add two sell limit orders
    s1 = Order('sell', 1, 5, price=101, order_type='LO')
    s2 = Order('sell', 2, 3, price=102, order_type='LO')
    ob.add_limit_order(s1)
    ob.add_limit_order(s2)

    # add a buy limit that should match partially with s1
    b1 = Order('buy', 3, 4, price=101, order_type='LO')
    trades = ob.add_limit_order(b1)
    assert len(trades) == 1
    maker_id, taker_id, price, qty = trades[0]
    assert price == 101
    assert qty == 4
    assert s1.remaining == 1

    # market buy to consume remaining sell at 101 and then 102
    m = Order('buy', 4, 5, order_type='MO')
    trades2 = ob.add_market_order(m)
    # should trade 1 at 101 and 3 at 102 (total 4), leaving market order with 1 remaining
    total = sum(t[3] for t in trades2)
    assert total == 4

    print('All simple tests passed')

def simple_test2():
    ob = OrderBook()
    # add two sell limit orders
    s1 = Order('sell', 1, 5, price=101, order_type='LO')
    s2 = Order('sell', 2, 3, price=102, order_type='LO')
    ob.add_limit_order(s1)
    ob.add_limit_order(s2)

    # add a buy limit that should match match with s1 and eat into s2
    b1 = Order('buy', 3, 6, price=102, order_type='LO')
    trades = ob.add_limit_order(b1)
    assert len(trades) == 2
    
    #Check if ask at 101 is fully consumed and ask at 102 is partially consumed, with correct prices and quantities.
    assert trades[0][2] == 101
    assert trades[0][3] == 5
    assert s1.remaining == 0

    assert trades[1][2] == 102
    assert trades[1][3] == 1
    assert s2.remaining == 2

    #Now check that we cannot self-trade by adding a buy order from the same trader as the existing sell orders, which should be skipped.
    b2 = Order('buy', 2, 2, price=102, order_type='LO')
    trades2 = ob.add_limit_order(b2)
    assert len(trades2) == 0

    print('All simple tests passed')

    
def orders_visual():
    # tiny self-test when run directly
    ob = OrderBook()
    all_trades = []
    '''
    Don't have to simulate here as a range, but as np.random for the prices, and then poisson for the quantity. Get mid price of order book and see how it evolves with time
    How can we involve time here?
    '''
    for i in range(90,105,1):
        trades = ob.add_limit_order(Order('buy', trader_id=1, quantity=poisson(5), price=i, order_type='LO'))
        all_trades.extend(trades)
    for i in range(95,110,1):
        trades = ob.add_limit_order(Order('sell', trader_id=2, quantity=poisson(5), price=i, order_type='LO'))
        all_trades.extend(trades)

    print('Snapshot:', ob.snapshot(depth=10))
    print('Trades:', all_trades)
    ob.display_order_book(10, 'Bar')
    ob.display_order_book(10, 'Curve')

if __name__ == '__main__':
    simple_test1()
    simple_test2()
    orders_visual()
