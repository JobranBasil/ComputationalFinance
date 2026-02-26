#Order Book Class
'''
Need to create: 
Order class with main features of order book, price, volume, etc.. 
OrderBook with all bids and asks with volumes.
Add function to allow traders to add limit order
Add function to create matching, when there is a bid that matches the ask, match them but based on what rules? Double auction, price time priority, research rules online.
Add function to market buys and market sells? Something like that.
'''


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.dirname(__file__), "ZI_OB_plots")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.random.seed(42)
simulation_steps = 100000
lambda_market = 7
lambda_limit = 10
lambda_cancel = 3


MAX_DEPTH = 10

best_bid = 100.0
best_ask = 101.0
order_book = {
    "bids": [(best_bid -0.01 * i, np.random.randint(1,10)) for i in range(5)],
    "asks": [(best_ask + 0.01 * i, np.random.randint(1, 10)) for i in range(5)],
}

fixed_snapshot = []

def enforce_max_depth(order_book):
    if len(order_book['bids']) > MAX_DEPTH:
        order_book['bids'] = order_book['bids'][:MAX_DEPTH]
    if len(order_book['asks']) > MAX_DEPTH:
        order_book['asks'] = order_book['asks'][:MAX_DEPTH]

def place_limit_order(order_type, price, quantity):
    if order_type == 'bid':
        for i, (p,q) in enumerate(order_book['bids']):
            if p == price:
                order_book['bids'][i] = (p, q + quantity)
                break
            else:
                order_book['bids'].append((price, quantity))
                order_book['bids'].sort(reverse=True)
    elif order_type == 'ask':
        for i, (p,q) in enumerate(order_book['asks']):
            if p == price:
                order_book['asks'][i] = (p, q + quantity)
                break
            else:
                order_book['asks'].append((price, quantity))
                order_book['asks'].sort()
    enforce_max_depth(order_book)

def update_best_prices():
    print('UPDATING BEST PRICE')
    global best_bid, best_ask
    if order_book['bids']:
        best_bid = max(order[0] for order in order_book['bids'])
    else:
        best_bid = 0.0
    if order_book['asks']:
        best_ask = min(order[0] for order in order_book['asks'])
    else:
        best_ask = float('inf')

def replenish_order_book_with_quantity():
    print('REPLENISHING OB')
    if len(order_book['bids']) < 5:
        print("REPLENISHING BIDS")
        for i in range(5 - len(order_book['bids'])):
            if np.random.rand()<0.5:
                print("REPLENISHING IN BEST BID")
                price= best_bid - 0.01 * (i + 1)
                quantity = np.random.randint(1, 10)
                place_limit_order('bid', price, quantity)
            else:
                pass
                #print("REPLENISHING OUT BEST BID")
                #price = best_bid + 0.01 * (i + 1)
                #quantity = np.random.randint(1, 10)
                #place_limit_order('bid', price, quantity)
    if len(order_book['asks']) < 5:
        print("REPLENISHING ASKS")
        for i in range(5 - len(order_book['asks'])):
            if np.random.rand()<0.5:
                print("REPLENISHING OUT BEST ASK")
                price= best_ask + 0.01 * (i + 1)
                quantity = np.random.randint(1, 10)
                place_limit_order('ask', price, quantity)
            else:
                pass
                #print("REPLENISHING IN BEST ASK")
                #price = best_ask - 0.01 * (i + 1)
                #quantity = np.random.randint(1, 10)
                #place_limit_order('ask', price, quantity)

def calculate_weighted_mid_price(order_book):
    total_bid_volume = sum([b[1] for b in order_book['bids']])
    total_ask_volume = sum([a[1] for a in order_book['asks']])

    weighted_bid_price = sum([b[0] * b[1] for b in order_book['bids']]) / total_bid_volume if total_bid_volume > 0 else 0
    weighted_ask_price = sum([a[0] * a[1] for a in order_book['asks']]) / total_ask_volume if total_ask_volume > 0 else float('inf')

    weighted_mid_price = (weighted_bid_price + weighted_ask_price)/ 2 if weighted_bid_price > 0 and weighted_ask_price < float('inf') else 0
    return weighted_mid_price



weighted_mid_prices=[]
spread_data = []
price_diffusion = []
event_types = []
bids= []
asks= []

for step in range(simulation_steps):
    print(f"BIDS LOG: {order_book['bids'][0]}")
    print(f"ASKS LOG: {order_book['asks'][0]}")

    replenish_order_book_with_quantity()

    event_type = np.random.choice(['market', 'limit', 'cancel'],
                                  p = [lambda_market, lambda_limit, lambda_cancel] /
                                  np.sum([lambda_market, lambda_limit, lambda_cancel]))

    if event_type == 'market':
        print("INCOMING MARKET ORDER")
        nq = np.random.randint(1, 5)
        if np.random.rand() < 0.5:
            print('MARKET BUY')# market buy
            if order_book['asks']:
                price, quantity = order_book['asks'][0]
                print(f"price: {price}, \n quantity: {quantity}")
                print(f' quantity take from best: {nq}')
                if quantity > 1 and quantity - nq >= 0:
                    order_book['asks'][0] = (price, quantity - nq)
                    print(order_book['asks'][0])
                    print(f'quantity at best: {quantity -nq}')
                elif quantity - nq == 0:
                    print(' market buy q - nq =0')
                    print(f"iterate: {order_book['asks'][0]}")
                    print(f'new quantity: {quantity}')
                    order_book['asks'].pop(0)
                    update_best_prices()
                else:
                    print('market buy q - nq  < 0')
                    print(f'quantity at best after full MO: {quantity - nq}')
                    print("EDGE")
                    nq2 = quantity - quantity
                    print(f'nq2: {nq2}')
                    remnq =  nq - quantity
                    print(f'remnq: {remnq}')
                    print(order_book['asks'])
                    order_book['asks'].pop(0)
                    price, quantity = order_book['asks'][0]
                    print(order_book['asks'])
                    print('level popped')
                    if quantity == 0:
                        order_book['asks'].pop(0)
                        price, quantity = order_book['asks'][0]
                    else:
                        pass
                    print('removing remaining MO q')
                    order_book['asks'][0] = (price, quantity - remnq)
                    print(f"best level ask: {order_book['asks'][0]}")
                    print('updating best ask level')
                    update_best_prices()
                    print('updated best ask level')
                    print(f"updated best level ask: {order_book['asks'][0]}")

        else: # markert sell
            if order_book['bids']:
                print('MARKET SELL')
                price, quantity = order_book['bids'][0]
                print(f"price: {price}, \n quantity: {quantity}")
                print(f' quantity take from best: {nq}')
                if quantity > 1 and quantity - nq >= 0:
                    order_book['bids'][0] = (price, quantity - nq)
                    print(order_book['bids'][0])
                    print(f'quantity at best: {quantity - nq}')
                elif quantity - nq == 0:
                    print(' market sell q - nq =0')
                    print(f"iterate: {order_book['bids'][0]}")
                    print(f'new quantity: {quantity}')
                    order_book['bids'].pop(0)
                    update_best_prices()
                else:
                    print('market sell q - nq  < 0')
                    print(f'quantity at best after full MO: {quantity - nq}')
                    print("EDGE")
                    nq2 = quantity - quantity
                    print(f'nq2: {nq2}')
                    remnq = nq - quantity
                    print(f'remnq: {remnq}')
                    print(order_book['bids'])
                    order_book['bids'].pop(0)
                    price, quantity = order_book['bids'][0]
                    print(order_book['bids'])
                    print('level popped')
                    if quantity == 0:
                        order_book['bids'].pop(0)
                        price, quantity = order_book['bids'][0]
                    else:
                        pass
                    print('removing remaining MO q')
                    order_book['bids'][0] = (price, quantity - remnq)
                    print(f"best level bid: {order_book['bids'][0]}")
                    print('updating best bid level')
                    update_best_prices()
                    print('updated best bid level')
                    print(f"updated best level bid: {order_book['bids'][0]}")

    elif event_type == 'limit':
        print('LIMIT INCOMING')
        price_shift = np.random.uniform(0.005, 0.01) # v tight limit order placement
        quantity = np.random.randint(1, 10)
        if np.random.rand() < 0.5: # limit buy
            print('LIMIT BUY')
            if quantity == 0:
                order_book['bids'].pop(0)
                price, quantity = order_book['bids'][0]
                place_limit_order('bid', best_bid - price_shift, quantity)
            else:
                place_limit_order('bid', best_bid - price_shift, quantity)

        else: #limit sell
            print('LIMIT SELL')
            if quantity == 0:
                order_book['asks'].pop(0)
                price, quantity = order_book['asks'][0]
                place_limit_order('ask', best_ask + price_shift, quantity)
            else:
                place_limit_order('ask', best_ask + price_shift, quantity)

    elif event_type == 'cancel':
        print('INCOMING CANCEL')
        if np.random.rand() < 0.5 and len(order_book['bids']) > 5: #cancel bid
            print("CANCEL BID")
            idx = np.random.randint(len(order_book['bids']))
            price, quantity = order_book['bids'][idx]
            if quantity > 1:
                order_book['bids'][idx] = (price, quantity - np.random.randint(1, quantity))
            else:
                order_book['bids'].pop(idx)
        elif len(order_book['asks']) > 5: # cance ask
            print("CANCEL ASK")
            idx = np.random.randint(len(order_book['asks']))
            price, quantity = order_book['asks'][idx]
            if quantity > 1:
                order_book['asks'][idx] = (price, quantity - np.random.randint(1, quantity))
            else:
                order_book['asks'].pop(idx)

    update_best_prices()

    if best_ask < float('inf') and best_bid > 0.0:
        bid = best_bid
        ask = best_ask
        spread = best_ask - best_bid
        mid_price = (best_ask + best_bid)/2
        weighted_mid_price = calculate_weighted_mid_price(order_book)
        bids.append(bid)
        asks.append(ask)
        spread_data.append(spread)
        price_diffusion.append(mid_price)
        weighted_mid_prices.append(weighted_mid_price)
        event_types.append(event_type)
    else:
        print('MARKET STRUCTURE AT INFINITE BOUNDS')

    if step % 1000 == 0:
        print(f'step {step}: Best bid={best_bid} , best ask={best_ask} , spread = {spread}')


    if step % 100 ==0:
        fixed_snapshot.append({
            'step': step,
            'bids': sorted(order_book['bids'], reverse=True),
            'asks': sorted(order_book['asks']),
            'event': event_type
        })

def visualise_order_book(snapshots):
    print('Visualising')
    num_snapshots = len(snapshots)
    fig, axes = plt.subplots(num_snapshots, 1, figsize=(80,80))

    if num_snapshots ==1:
        axes=[axes]

    for i, snapshot in enumerate(snapshots):
        step = snapshot['step']
        bids = snapshot['bids']
        asks = snapshot['asks']
        event = snapshot['event']
        ax = axes[i]

        bid_prices = [b[0] for b in bids]
        bid_quantities = [b[1] for b in bids]
        ask_prices = [a[0] for a in asks]
        ask_quantities = [a[1] for a in asks]

        price_spacing = 0.075
        spaced_bid_prices = [p - i * price_spacing for i, p in enumerate(bid_prices)]
        spaced_ask_prices = [p + i * price_spacing for i, p in enumerate(ask_prices)]

        ax.bar(spaced_bid_prices, bid_quantities, width=0.001, color='green', label ='Bids')
        ax.bar(spaced_ask_prices, [-q for q in ask_quantities], width=0.001, color='red', label ='Asks')

        if bid_prices and ask_prices:
            best_bid = max(bid_prices)
            best_ask = min(ask_prices)
            mid_price = (best_bid + best_ask)/2
            spread = best_ask - best_bid

            ax.axvline(best_bid, color='blue', linestyle='--', label='Best Bid', alpha=0.2)
            ax.axvline(best_ask, color='orange', linestyle='--', label='Best Ask', alpha=0.2)
            ax.axvline(mid_price, color='black', linestyle='--', label='Mid price')

            ax.text(mid_price, 0, f'Spread: {spread:.4f}', color = 'black', ha='center', va='bottom')

        ax.set_title(f'Step {step}  | Event: {event.capitalize()}')
        ax.set_xlabel('price')
        ax.set_ylabel('Quantity')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    #plt.tight_layout()
    plot_path = os.path.join(output_dir, "Simulated_Order_Book5.png")
    plt.savefig(plot_path)
    plt.close()


visualise_order_book(fixed_snapshot[:10])


depth_validation = [(snap['step'], len(snap['bids']), len(snap['asks'])) for snap in fixed_snapshot]
print(depth_validation[:5])

results = pd.DataFrame({
    'Spread': spread_data,
    'Mid Price': price_diffusion,
    'Weighted Mid Price': weighted_mid_prices,
    'Event':event_types,
    'Best Bid':bids,
    'Best Ask': asks
}).dropna()




plt.figure(figsize=(10,10))
plt.plot(results['Mid Price'], label='Mid-Price diffusion')
plt.xlabel('time step')
plt.ylabel('Mid Price')
plt.title('mid price diffusion')
plt.legend()
plot_path = os.path.join(output_dir, "mid_price_diffusion5.png")
plt.savefig(plot_path)
plt.close()

plt.figure(figsize=(10,10))
plt.plot(results['Spread'], label='Bid ask spread')
plt.xlabel('time step')
plt.ylabel('Spread')
plt.title('Bid ask spreads ')
plt.legend()
plot_path = os.path.join(output_dir, "Spreads5.png")
plt.savefig(plot_path)
plt.close()

plt.figure(figsize=(10,10))
plt.plot(results.index, results['Mid Price'], label='Mid-Price diffusion', alpha=0.7)
plt.plot(results.index, results['Weighted Mid Price'], label='Weighted Mid-Price', linestyle='--', alpha=0.7)
plt.xlabel('time step')
plt.ylabel('Price')
plt.title('mid price and weighted mid price diffusion')
plt.legend()
plt.grid(True)
plot_path = os.path.join(output_dir, "weighted_and_mid_price_diffusion5.png")
plt.savefig(plot_path)
plt.close()

plt.figure(figsize=(10,10))
plt.plot(results.index, results['Mid Price'], label='Mid-Price diffusion', alpha=0.7)
plt.plot(results.index, results['Spread'], label='Bid Ask spread', linestyle='--', alpha=0.7)
plt.xlabel('time step')
plt.ylabel('Price / Spread')
plt.title('mid price diffusion with Spread overlay')
plt.legend()
plt.grid(True)
plot_path = os.path.join(output_dir, "mid_price_diffusion_spread_overlay5.png")
plt.savefig(plot_path)
plt.close()


