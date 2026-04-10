import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from src.visualization import figureStyle


def run_backtest(BARS, strategy, initial_cash=100_000.0, **strategy_kwargs):
    """
    Simulate a strategy over historical BARS.
    Orders placed on bar i are checked for fills starting on bar i+1 (no lookahead).
    Limit buy fills when bar Low <= limit_price; limit sell fills when bar High >= limit_price.

    Returns:
        trade_log    — list of fill dicts: {timestamp, side, qty, price}
        equity_curve — pd.Series of portfolio value after each bar
    """
    cash    = initial_cash
    qty     = 0.0
    pending = []   # list of {side, qty, limit_price}
    trade_log   = []
    equity_vals = []

    for i in range(1, len(BARS)):
        bar = BARS.iloc[i]

        # 1. Check pending GTC (Good Til Canceled) limit orders against this bar
        remaining = []
        for order in pending:
            if order['side'] == 'buy' and bar['Low'] <= order['limit_price']:
                cost = order['limit_price'] * order['qty']
                if cash >= cost:
                    cash -= cost
                    qty  += order['qty']
                    trade_log.append({'timestamp': BARS.index[i], 'side': 'buy', 'qty': order['qty'], 'price': order['limit_price']})
                else:
                    remaining.append(order)  # can't afford — keep pending
            elif order['side'] == 'sell' and bar['High'] >= order['limit_price']:
                sell_qty = min(order['qty'], qty)
                if sell_qty > 0:
                    cash += order['limit_price'] * sell_qty
                    qty  -= sell_qty
                    trade_log.append({'timestamp': BARS.index[i], 'side': 'sell', 'qty': sell_qty, 'price': order['limit_price']})
            else:
                remaining.append(order)
        pending = remaining

        # 2. Run strategy on all bars up to and including bar i
        order_info = strategy(BARS.iloc[:i + 1], **strategy_kwargs)
        if order_info['move'] in ('buy', 'sell'):
            pending.append({
                'side':        order_info['move'],
                'qty':         order_info['qty'],
                'limit_price': order_info['limit_price'],
            })

        # 3. Record portfolio value
        equity_vals.append(cash + qty * bar['Close'])

    equity_curve = pd.Series(equity_vals, index=BARS.index[1:], name='equity')
    return trade_log, equity_curve


def compute_metrics(trade_log, equity_curve, initial_cash):
    final_value  = equity_curve.iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    rolling_max  = equity_curve.cummax()
    max_drawdown = ((equity_curve - rolling_max) / rolling_max).min() * 100

    bar_returns  = equity_curve.pct_change().dropna()
    sharpe       = (bar_returns.mean() / bar_returns.std()
                    if bar_returns.std() > 0 else 0.0)

    # Win rate: pair buys and sells FIFO into round trips
    buys  = [t['price'] for t in trade_log if t['side'] == 'buy']
    sells = [t['price'] for t in trade_log if t['side'] == 'sell']
    pairs = list(zip(buys, sells))
    win_rate = (sum(1 for b, s in pairs if s > b) / len(pairs) * 100) if pairs else 0.0

    return {
        'total_return_pct': round(total_return, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'sharpe_ratio':     round(sharpe, 3),
        'win_rate_pct':     round(win_rate, 1),
        'n_trades':         len(trade_log),
        'final_value':      round(final_value, 2),
    }


def plotBacktest(BARS, trade_log, equity_curve, metrics, symbol='BTC/USD', strategy_name=''):
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
    for trade in trade_log:
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
    ax3.plot(range(len(equity_curve)), equity_curve.values, color='#58a6ff', linewidth=1)
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
