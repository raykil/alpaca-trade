def momentum(BARS, window=3, threshold=150, qty=0.001):
    avgPrices = BARS['avgPrice'].to_numpy()
    current_price = float(avgPrices[-1])
    move = 'hold'
    if len(avgPrices) >= window:
        delta_p = avgPrices[-1] - avgPrices[-window]
        if   (delta_p >  threshold): move = 'buy'
        elif (delta_p < -threshold): move = 'sell'
    return {'move': move, 'qty': qty, 'limit_price': current_price}

def reverse_momentum(BARS, window=5, threshold=150, qty=0.001):
    avgPrices = BARS['avgPrice'].to_numpy()
    current_price = float(avgPrices[-1])
    move = 'hold'
    if len(avgPrices) >= window:
        delta_p = avgPrices[-1] - avgPrices[-window]
        if   (delta_p >  threshold): move = 'sell'
        elif (delta_p < -threshold): move = 'buy'
    return {'move': move, 'qty': qty, 'limit_price': current_price}

def order_test(BARS, qty=0.001):
    """
    This is not an actual strategy. Just to make sure orders are always made.
    If price went down, buy. If went up, sell.
    """
    avgPrices = BARS['avgPrice'].to_numpy()
    current_price = float(avgPrices[-1])
    move = 'hold'
    if   avgPrices[-2] < avgPrices[-1]: move = 'sell'
    elif avgPrices[-2] > avgPrices[-1]: move = 'buy'
    return {'move': move, 'qty': qty, 'limit_price': current_price}

def momentum_v2(BARS, window=3, threshold=150, qty=0.001):
    """
    Like momentum, but returns a dict with order fields instead of just a move string.
    - limit_price: current avgPrice (the price you're willing to buy/sell at)
    - qty: quantity to order
    """
    avgPrices = BARS['avgPrice'].to_numpy()
    current_price = float(avgPrices[-1])
    move = 'hold'
    if len(avgPrices) >= window:
        delta_p = avgPrices[-1] - avgPrices[-window]
        if   delta_p >  threshold: move = 'buy'
        elif delta_p < -threshold: move = 'sell'
    return {'move': move, 'qty': qty, 'limit_price': current_price}

strategy_map = {
    'momentum':         momentum,
    'momentum_v2':      momentum_v2,
    'reverse_momentum': reverse_momentum,
    'order_test':       order_test,
}
