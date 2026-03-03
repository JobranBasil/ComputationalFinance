from collections import deque, defaultdict
import itertools
import time
import matplotlib.pyplot as plt

plt.style.use('ggplot')

class Order:
    
    """Simple order representation.

    Attributes:
        id: unique integer id
        side: 'buy' or 'sell'
        price: float or None for market orders
        quantity: original quantity
        remaining: remaining quantity to fill
        order_type: 'limit' or 'market' (Abbreviated as 'LO' and 'MO')
        timestamp: creation time (float)
    """

    _ids = itertools.count(1)

    def __init__(self, side, trader_id, quantity, price=None, order_type='LO', timestamp=None):
        assert side in ('buy', 'sell')
        assert order_type in ('LO', 'MO')
        if order_type == 'LO':
            assert price is not None

        self.oid = next(Order._ids)
        self.tid = trader_id # trader id, to be set when order is placed by an agent
        self.side = side
        self.price = price
        self.quantity = float(quantity)
        self.remaining = float(quantity)
        self.order_type = order_type
        self.timestamp = time.time() if timestamp is None else timestamp #Record time of order creation, to allow for flexibility when matching orders.

    def __repr__(self):
        return f"Order(id={self.oid}, tid={self.tid}, {self.side}, {self.order_type}, price={self.price}, rem={self.remaining})"

class OrderBook:

    #Store bids and asks in a dictionary, with bids in descending order and asks in ascending order.
        
    def __init__(self):
        #Create dictionaries to store bids and asks, where the key is the price and the value is a deque of orders at that price level.
        self.bids = defaultdict(deque)
        self.asks = defaultdict(deque)
        # maintain sorted list of prices for quick best-price lookup (small book, kept simple)
        self._bid_prices = set()
        self._ask_prices = set()

    def best_bid(self):
        #Create function to get best bid price. If there are no bids, return None.
        if not self._bid_prices:
            return None
        return max(self._bid_prices)
        
    def best_ask(self):
        #Create function to get best ask price. If there are no asks, return None.
        if not self._ask_prices:
            return None
        return min(self._ask_prices)
        
    def _match_at_price(self, incoming: Order, price: float):
        """
        Match incoming order against orders at given price level.
        Returns list of trade tuples: (maker_order_id, taker_order_id, price, quantity)
        """
        trades = []
        #If someone wants to buy, get the queue of asks. If someone wants to sell, get the queue of bids.
        book_side = self.asks if incoming.side == 'buy' else self.bids
        #Get the deque of orders at the given price level.
        price_deque = book_side[price]

        while incoming.remaining > 0 and price_deque:
            #Get first order in the deque, which is the oldest order at that price level.
            maker = price_deque[0] 
            if maker.tid == incoming.tid:
                #If the maker order is from the same trader as the incoming order, skip it to avoid self-trading.
                price_deque.popleft() #Remove the maker order from the deque and continue to the next one.
                if not price_deque:
                    #If there are no more orders at that price level, remove the price level from the book.
                    if incoming.side == 'buy':
                        # we consumed asks
                        self._ask_prices.discard(price)
                        del self.asks[price]
                    else:
                        self._bid_prices.discard(price)
                        del self.bids[price]
                continue
            else:
                #Calculated the quantity to trade, minimum of the quantity of order and what is available.
                traded_qty = min(incoming.remaining, maker.remaining)
                #Update reamining quantity left by taking away the traded quantity from both the incoming order and the maker order.
                incoming.remaining -= traded_qty
                maker.remaining -= traded_qty
                #Add trade to list of trades, with maker order id, incoming order id, price, and traded quantity.
                trades.append((maker.tid, incoming.tid, price, traded_qty))

            #Remove maker order from deque if fully filled.
            if maker.remaining == 0:
                price_deque.popleft()
            if not price_deque:
                # remove price level
                if incoming.side == 'buy':
                    # we consumed asks
                    self._ask_prices.discard(price)
                    del self.asks[price]
                else:
                    self._bid_prices.discard(price)
                    del self.bids[price]

        return trades
        
    def add_limit_order(self, order: Order):
        '''
        Ensure that the order being added is a limit order. 
        If it is a buy order, attempt to match it against the best ask price. If it is a sell order, attempt to match it against the best bid price. 
        If the order cannot be fully filled, add the remaining quantity to the appropriate side of the book.
        '''
        assert order.order_type == 'LO'
        trades = []
        if order.side == 'buy': #Buy order
            # try match with asks while price >= best ask
            while order.remaining > 0:
                #Find best ask price. If there are no asks, break out of the loop.
                best_ask = self.best_ask()
                if best_ask is None or best_ask > order.price:
                    '''
                    If the best ask price is more than the order price, then the order cannot be filled at the current best price, 
                    so break out of the loop and add to trades
                    '''
                    break
                #Add trade to list
                trades += self._match_at_price(order, best_ask)
            if order.remaining > 0:
                #If there is remaining quantity, add it to the bids dictionary and update the set of bid prices.
                self.bids[order.price].append(order)
                self._bid_prices.add(order.price)
        else:  # sell order
            # try match with bids while price <= best bid
            while order.remaining > 0:
                #Find best bid price. If there are no bids, break out of the loop.
                best_bid = self.best_bid()
                '''
                If the best bid price is less than the order price, then the order cannot be filled at the current best price, 
                so break out of the loop and add to trades
                '''
                if best_bid is None or best_bid < order.price:
                    break
                #Add trade to list
                trades += self._match_at_price(order, best_bid)
            if order.remaining > 0:
                #If there is reamining quantity, add it to asks dictionary at that price and update set of ask prices.
                self.asks[order.price].append(order)
                self._ask_prices.add(order.price)

        return trades
        
    def add_market_order(self, order: Order):
        #Ensure that the order being added is a market order.
        assert order.order_type == 'MO'
        #Create empty trades list
        trades = []
        #If we are trying to buy something, as a market order, we want to get the best ask and match the buy to the lowest ask price until the order is fully filled.
        if order.side == 'buy': 
            #For buy order, attempt to match it against the best ask price until the order is fully filled or there are no more asks.
            while order.remaining > 0:
                best_ask = self.best_ask()
                if best_ask is None:
                    break
                #Add market trade to list
                trades += self._match_at_price(order, best_ask)
        else:
            #For sell order, attempt to match it against the best bid price until the order is fully filled or there are no more bids.
            while order.remaining > 0:
                best_bid = self.best_bid()
                if best_bid is None:
                    break
                #Add market trade to list
                trades += self._match_at_price(order, best_bid)
        return trades
        
    def top_of_book(self):

        '''
        Return the top of the book, which is the best bid and best ask price, just to see what the current state of the book is.
        '''

        return {
            'best_bid': self.best_bid(),
            'best_ask': self.best_ask()
        }

    def snapshot(self, depth=5):

        '''
        Return a snapshot of the order book at a given depth specified by the user.
        Default depth is 5, returning 5 bid and ask prices from either side of the book.
        '''

        bids = sorted(self._bid_prices, reverse=True)[:depth]
        asks = sorted(self._ask_prices)[:depth]

        return {
            'bids': [(p, sum(o.remaining for o in self.bids[p])) for p in bids],
            'asks': [(p, sum(o.remaining for o in self.asks[p])) for p in asks]
        }
    
    def display_order_book(self, depth, type):

        #We either want a bar chart showing the levels of the book, or a demand curve showing cumulative quantity at each price level.
        #We can use snapshot function to get the current state of the book, and then use matplotlib to create the desired visualization.
        assert type in ('Bar', 'Curve')

        snapshot = self.snapshot(depth)
        bids = snapshot['bids']
        asks = snapshot['asks']

        if type == 'Bar':
            #Create bar chart of the book, with bid prices on the left and ask prices on the right.
            plt.bar([p for p, q in bids], [q for p, q in bids], color='blue', label='Bids')
            plt.bar([p for p, q in asks], [q for p, q in asks], color='red', label='Asks')
            plt.xlabel('Price')
            plt.ylabel('Quantity')
            plt.title('Order Book')
            plt.legend()
            plt.show()
        else:
            #Create demand curve of the book, with cumulative quantity at each price level.
            bid_prices = [p for p, q in bids]
            bid_quantities = [sum(q for p, q in bids if p >= price) for price in bid_prices]

            ask_prices = [p for p, q in asks]
            ask_quantities = [sum(q for p, q in asks if p <= price) for price in ask_prices]

            # Piecewise constant (step) plots
            plt.step(bid_quantities, bid_prices, where='post', color='blue', label='Bids')
            plt.step(ask_quantities, ask_prices, where='post', color='red', label='Asks')

            plt.xlabel('Cumulative Quantity')
            plt.ylabel('Price')
            plt.title('Demand Curve')
            plt.legend()
            plt.show()


