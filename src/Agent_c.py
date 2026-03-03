from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
import numpy as np
from OrderBook import Order, OrderBook

class Trader:

    _ids = itertools.count(1)

    def __init__(self):
        self.tid = next(Trader._ids)

    def decide(self, orderbook):
        return None
    
class NoiseTrader(Trader):

    def decide(self, orderbook):
        #Randomly decide to buy or sell with equal probability, and randomly choose a price and quantity.
        side = np.random.choice(['buy', 'sell'])
        price = np.random.uniform(101,105) #Random price between min and max price in orderbook
        quantity = np.random.poisson(5) #Random quantity between 1 and 10
        
        return Order(side, self.tid, quantity, price=price, order_type='LO')
    
if __name__ == '__main__':
    #Test the NoiseTrader by creating an instance and calling the decide method with a sample order book.
    ob = OrderBook()
    for i in range(10):
        trader = NoiseTrader()
        order = trader.decide(ob)
        print(order)
        ob.add_limit_order(order)
    print(ob.snapshot(depth=10))