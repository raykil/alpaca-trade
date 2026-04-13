import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from src.visualization import figureStyle

def OrderBuy(order, bar_index, BARS, cash, quantity, tradeLog, remainingOrders):
    cost = order['limit_price'] * order['qty']
    if cash >= cost:
        cash -= cost
        quantity += order['qty']
        tradeLog.append({ # assumes all order is filled exactly at limit price (conservative, which is good) at the moment, all or nothing. It's good assumption for now.
            'timestamp': BARS.index[bar_index], 
            'side': 'buy', 
            'qty': order['qty'], 
            'price': order['limit_price']
        })
    else:
        remainingOrders.append(order)  # can't afford — keep pending
    return cash, quantity

def OrderSell(order, bar_index, BARS, cash, quantity, tradeLog):
    sell_qty = min(order['qty'], quantity)
    if sell_qty > 0:
        cash += order['limit_price'] * sell_qty
        quantity -= sell_qty
        tradeLog.append({
            'timestamp': BARS.index[bar_index], 
            'side': 'sell', 
            'qty': sell_qty, 
            'price': order['limit_price']
        })
    return cash, quantity

def run_backtest(BARS, strategy, initial_cash=100_000.0, **strategy_kwargs):
    # Initialize assets
    cash          = initial_cash
    quantity      = 0.0
    pendingOrders = [] # list of {side, qty, limit_price}. Limit price: execute order iff the price I suggested is better.
    tradeLog      = []
    equityValue   = []

    for i in range(1, len(BARS)):
        bar = BARS.iloc[i]

        # Check pending orders
        remainingOrders = []
        for order in pendingOrders:
            goodToBuy  = order['side'] == 'buy'  and bar['Low']  <= order['limit_price']
            goodToSell = order['side'] == 'sell' and bar['High'] >= order['limit_price']
            if    goodToBuy : cash, quantity = OrderBuy(order, i, BARS, cash, quantity, tradeLog, remainingOrders)
            elif  goodToSell: cash, quantity = OrderSell(order, i, BARS, cash, quantity, tradeLog)
            else: remainingOrders.append(order)
        pendingOrders = remainingOrders

        # Run strategy and make orders if appropriate
        orderInfo = strategy(BARS.iloc[:i + 1], **strategy_kwargs)
        if orderInfo['move'] in ('buy', 'sell'):
            pendingOrders.append({
                'side':        orderInfo['move'],
                'qty':         orderInfo['qty'],
                'limit_price': orderInfo['limit_price'],
            })

        # Record portfolio value
        equityValue.append(cash + quantity * bar['Close'])

    equityCurve = pd.Series(equityValue, index=BARS.index[1:], name='equity')
    return tradeLog, equityCurve


def compute_metrics(tradeLog, equityCurve, initial_cash):
    final_value  = equityCurve.iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    rolling_max  = equityCurve.cummax()
    max_drawdown = ((equityCurve - rolling_max) / rolling_max).min() * 100

    bar_returns  = equityCurve.pct_change().dropna()
    sharpe       = (bar_returns.mean() / bar_returns.std()
                    if bar_returns.std() > 0 else 0.0)

    # Win rate: pair buys and sells FIFO into round trips
    buys  = [t['price'] for t in tradeLog if t['side'] == 'buy']
    sells = [t['price'] for t in tradeLog if t['side'] == 'sell']
    pairs = list(zip(buys, sells))
    win_rate = (sum(1 for b, s in pairs if s > b) / len(pairs) * 100) if pairs else 0.0

    return {
        'total_return_pct': round(total_return, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'sharpe_ratio':     round(sharpe, 3),
        'win_rate_pct':     round(win_rate, 1),
        'n_trades':         len(tradeLog),
        'final_value':      round(final_value, 2),
    }


def plotBacktest(BARS, tradeLog, equityCurve, metrics, symbol='BTC/USD', strategy_name=''):
    style = figureStyle()
    fig = mpf.figure(style=style, figsize=(16, 11))
    fig.suptitle(f"{symbol}  —  {strategy_name}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.4)

    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax2.set_xlabel('')
    ax3.set_xlabel('Bar')

    mpf.plot(BARS, ax=ax1, volume=ax2, type='candle', style=style)

    # Trade markers
    for trade in tradeLog:
        ts = trade['timestamp']
        if ts not in BARS.index:
            continue
        x      = BARS.index.get_loc(ts)
        is_buy = trade['side'] == 'buy'
        ax1.plot(x, trade['price'],
                 marker='^' if is_buy else 'v',
                 color='#00e676' if is_buy else '#ff5252',
                 markersize=8, zorder=5, linestyle='None')

    buy_handle  = mlines.Line2D([], [], marker='^', color='#00e676', linestyle='None', markersize=8, label='Buy fill')
    sell_handle = mlines.Line2D([], [], marker='v', color='#ff5252', linestyle='None', markersize=8, label='Sell fill')
    ax1.legend(handles=[buy_handle, sell_handle],
               loc='upper left', framealpha=0.6,
               facecolor='#22272d', edgecolor='#39424c', labelcolor='whitesmoke')

    # Equity curve
    ax3.plot(range(len(equityCurve)), equityCurve.values, color='#58a6ff', linewidth=1)
    ax3.set_ylabel('Portfolio $', color='whitesmoke', fontsize=10)
    ax3.tick_params(colors='whitesmoke')
    ax3.set_facecolor('#22272d')
    ax3.grid(True, linestyle='--', color='#39424c', linewidth=0.5)

    # Metrics bar at the bottom
    m = metrics
    metrics_str = (f"Return: {m['total_return_pct']:+.2f}%    "
                   f"Max drawdown: {m['max_drawdown_pct']:.2f}%    "
                   f"Sharpe: {m['sharpe_ratio']:.3f}    "
                   f"Win rate: {m['win_rate_pct']:.1f}%    "
                   f"Trades: {m['n_trades']}    "
                   f"Final: ${m['final_value']:,.2f}")
    fig.text(0.5, 0.01, metrics_str, ha='center', color='whitesmoke', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#22272d', edgecolor='#39424c', alpha=0.9))

    plt.show()
