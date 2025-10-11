# all strategies return a string "buy", "sell", or a NoneType.
"""
Order Types: https://docs.alpaca.markets/docs/orders-at-alpaca#order-types
Bracket Order: https://docs.alpaca.markets/docs/orders-at-alpaca#bracket-orders
"""

def momentum_strategy(PriceHistory, timeWindow: float) -> str:
    """
    timeWindow: how many seconds in the past to look into.
    If midPrice starts to rise by a threshold amount, the strategy returns 'buy'.
    If midPrice starts to fall by a threshold amount, the strategy returns 'sell'.

    TODO: How to detect better signals:
    - Layer a volatility-adaptive filter on your delta (e.g., require delta > k·sigma(return) over the window, smooth price with an EMA, and ignore signals when spread is wide or the book is thin) to cut noise.
    - Add confirmations: require N consecutive upticks, price above a short EMA/VWAP, rising volume or quote-imbalance, and avoid overbought/oversold extremes with RSI/Stochastic or by demanding “higher high + higher low” patterns.
    - Use a regime gate (e.g., trend strength via ADX or rolling R² of a price regression) and pair the signal with risk controls—tight stop, asymmetric take-profit, time stop, and skip trades around macro events—to trade momentum only when the market is actually trending.
    """
    buy_threshold = +0.06
    sel_threshold = -0.06
    while PriceHistory[-1][1] - PriceHistory[0][1] > timeWindow:
        PriceHistory.popleft()
    mp_i, mp_f = PriceHistory[0][0], PriceHistory[-1][0]
    delta_mp = mp_f - mp_i
    if   delta_mp > buy_threshold: move = 'buy'
    elif delta_mp < sel_threshold: move = 'sell'
    else: move = None
    print(f"i: {PriceHistory[-1][0]} f: {PriceHistory[0][0]}, delta: {round(PriceHistory[-1][0] - PriceHistory[0][0], 3)} move: {move}")
    return move


def makePayload(Q: dict, side: str) -> dict: # entryPrice = quote['midPrice']
    takeProf_offset = 0.20
    stopLoss_offset = 0.20

    if side=="buy":
        takeProf_price = Q['askPrice'] + takeProf_offset
        stopLoss_price = Q['askPrice'] - stopLoss_offset
    elif side=="sell":
        takeProf_price = Q['bidPrice'] - takeProf_offset
        stopLoss_price = Q['bidPrice'] + stopLoss_offset

    payload = {
        "side"  : side,
        "symbol": "AAPL",
        "type"  : "limit",
        "qty"   : 3,
        "time_in_force": "day",
        "order_class"  : "bracket",
        "limit_price"  : f"{Q['midPrice']:.2f}",
        "take_profit"  : {"limit_price": f"{takeProf_price:.2f}"},
        "stop_loss"    : {"stop_price" : f"{stopLoss_price:.2f}"},
    }
    return payload



STRATEGIES = {
    "momentum": {"func": momentum_strategy, "args": [0.15]}, # args: [timeWindow(seconds)]
}