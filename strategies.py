import numpy as np
import inspect

# TODO: I want to write strategies in simple way, i.e., returns single move dict.
# But then backtest time gets way slower. Haven't found solution yet.

def momentum_simple(avgPrices, window=5, threshold=0.001, qty=0.001):
    current_price = float(avgPrices[-1])
    move = 'hold'
    if len(avgPrices) > window:
        delta_p = (avgPrices[-1] - avgPrices[-window-1]) / avgPrices[-window-1]
        if   (delta_p >  threshold): move = 'buy'
        elif (delta_p < -threshold): move = 'sell'
    return {'move': move, 'qty': qty, 'limit_price': current_price} 

def momentum(BARS, window=3, threshold=0.001, qty=0.001):
    avgPrices = BARS['avgPrice'].to_numpy()
    avgPrices_f  = avgPrices[window:]
    avgPrices_i  = avgPrices[:-window]
    percent_diffs = (avgPrices_f - avgPrices_i)/avgPrices_i # how much percent change relative to initial price in the window (- means price went down)
    percent_diffs = np.pad(percent_diffs, (window, 0), constant_values=np.nan)

    MOVES = []
    for idx in range(1, len(avgPrices)):
        pct_diff = percent_diffs[idx]
        if np.isnan(pct_diff)     : move = 'hold'
        elif pct_diff >  threshold: move = 'buy'
        elif pct_diff < -threshold: move = 'sell'
        else                      : move = 'hold'
        MOVES.append({'move': move, 'qty': qty, 'limit_price': avgPrices[idx]})

    return MOVES

def reverse_momentum_simple(avgPrices, window=5, threshold=0.001, qty=0.001):
    current_price = float(avgPrices[-1])
    move = 'hold'
    if len(avgPrices) > window:
        delta_p = (avgPrices[-1] - avgPrices[-window-1]) / avgPrices[-window-1]
        if   (delta_p >  threshold): move = 'sell'
        elif (delta_p < -threshold): move = 'buy'
    return {'move': move, 'qty': qty, 'limit_price': current_price} 

def reverse_momentum(BARS, window=3, threshold=0.001, qty=0.001):
    avgPrices = BARS['avgPrice'].to_numpy()
    avgPrices_f  = avgPrices[window:]
    avgPrices_i  = avgPrices[:-window]
    percent_diffs = (avgPrices_f - avgPrices_i)/avgPrices_i # how much percent change relative to initial price in the window (- means price went down)
    percent_diffs = np.pad(percent_diffs, (window, 0), constant_values=np.nan)

    MOVES = []
    for idx in range(1, len(avgPrices)):
        pct_diff = percent_diffs[idx]
        if np.isnan(pct_diff)     : move = 'hold'
        elif pct_diff >  threshold: move = 'sell'
        elif pct_diff < -threshold: move = 'buy'
        else                      : move = 'hold'
        MOVES.append({'move': move, 'qty': qty, 'limit_price': avgPrices[idx]})

    return MOVES

def rsi_bb_reversion_simple(avgPrices, rsi_period=14, bb_period=20, bb_std=2.0,
                            rsi_oversold=30, rsi_overbought=70, qty=0.001):
    current_price = float(avgPrices[-1])

    if len(avgPrices) < max(rsi_period, bb_period) + 1:
        return {'move': 'hold', 'qty': qty, 'limit_price': current_price}

    # RSI — Wilder smoothing (manual EWM)
    alpha = 1 / rsi_period
    delta = np.diff(avgPrices)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain, avg_loss = gains[0], losses[0]
    for g, l in zip(gains[1:], losses[1:]):
        avg_gain = alpha * g + (1 - alpha) * avg_gain
        avg_loss = alpha * l + (1 - alpha) * avg_loss
    cur_rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss != 0 else 100.0

    # Bollinger Bands (last bb_period bars)
    window = avgPrices[-bb_period:]
    mean   = window.mean()
    std    = window.std(ddof=1)
    upper  = mean + bb_std * std
    lower  = mean - bb_std * std

    move = 'hold'
    if   cur_rsi < rsi_oversold  and current_price <= lower: move = 'buy'
    elif cur_rsi > rsi_overbought and current_price >= upper: move = 'sell'

    return {'move': move, 'qty': qty, 'limit_price': current_price}

def rsi_bb_reversion(BARS, rsi_period=14, bb_period=20, bb_std=2.0,
                              rsi_oversold=30, rsi_overbought=70, qty=0.001):
    # Compute RSI and Bollinger Bands once for the entire series — O(n)
    prices = BARS['avgPrice']

    delta = prices.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rsi   = (100 - 100 / (1 + gain / loss)).to_numpy()

    sma   = prices.rolling(bb_period).mean().to_numpy()
    std_v = prices.rolling(bb_period).std().to_numpy()
    upper = sma + bb_std * std_v
    lower = sma - bb_std * std_v
    prices_arr = prices.to_numpy()

    min_period = max(rsi_period, bb_period) + 1
    result = []
    for i in range(1, len(prices_arr)):
        p = float(prices_arr[i])
        if i < min_period or np.isnan(rsi[i]):
            move = 'hold'
        elif rsi[i] < rsi_oversold  and p <= lower[i]: move = 'buy'
        elif rsi[i] > rsi_overbought and p >= upper[i]: move = 'sell'
        else: move = 'hold'
        result.append({'move': move, 'qty': qty, 'limit_price': p})
    return result

strategy_map = {
    'momentum':                  momentum,
    'reverse_momentum':          reverse_momentum,
    'reverse_momentum_simple':   reverse_momentum_simple,
    'rsi_bb_reversion':          rsi_bb_reversion,
    'rsi_bb_reversion_simple':   rsi_bb_reversion_simple,
}

def liveTrade(strategy, BARS, **kwargs):
    window = kwargs.get('window', inspect.signature(strategy).parameters['window'].default)
    return strategy(BARS.iloc[-window:], **kwargs)[-1]

def prettyPrint(arr, howmany=None):
    var_name = next((k for k, v in inspect.currentframe().f_back.f_locals.items() if v is arr), "arr")
    rounded = [round(x, 6) for x in arr.tolist()]
    print(f"{var_name}: {rounded[:howmany]}")