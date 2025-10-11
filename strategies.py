# all strategies return a string "buy", "sel", or a NoneType.
"""
Order Types: https://docs.alpaca.markets/docs/orders-at-alpaca#order-types
Bracket Order: https://docs.alpaca.markets/docs/orders-at-alpaca#bracket-orders
"""

def momentum_strategy(PriceHistory, timeWindow: float) -> str:
    """
    timeWindow: how many seconds in the past to look into.
    If midPrice starts to rise by a threshold amount, the strategy returns 'buy'.
    If midPrice starts to fall by a threshold amount, the strategy returns 'sel'.
    """
    buy_threshold = 0.06
    sel_threshold = -0.06
    while PriceHistory[-1][1] - PriceHistory[0][1] > timeWindow:
        PriceHistory.popleft()
    mp_i, mp_f = PriceHistory[0][0], PriceHistory[-1][0]
    delta_mp = mp_f - mp_i
    if   delta_mp > buy_threshold: move = 'buy'
    elif delta_mp < sel_threshold: move = 'sel'
    else: move = None
    print(PriceHistory)
    return move



def makePayload(entryPrice: float, side: str) -> dict: # entryPrice = quote['midPrice']
    side = "buy" # "sell"
    takeProf_offset = 0.20
    stopLoss_offset = 0.20

    if side=="buy":
        takeProf_price = entryPrice + takeProf_offset
        stopLoss_price = entryPrice + stopLoss_offset
    elif side=="sell":
        takeProf_price = entryPrice - takeProf_offset
        stopLoss_price = entryPrice - stopLoss_offset

    payload = {
        "side"  : side,
        "symbol": "AAPL",
        "type"  : "limit",
        "qty"   : 3,
        "time_in_force": "day",
        "order_class"  : "bracket",
        "limit_price"  : f"{entryPrice:.2f}",
        "take_profit"  : {"limit_price": f"{takeProf_price:.2f}"},
        "stop_loss"    : {"stop_price" : f"{stopLoss_price:.2f}"},
    }
    return payload


STRATEGIES = {
    "momentum": {"func": momentum_strategy, "args": [0.15]}, # args: [timeWindow(seconds)]
}