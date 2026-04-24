import os, matplotlib
from contextlib import contextmanager
import pandas as pd
import quantstats as qs

# ── colour palette ───────────────────────────────────────────────────────────
_BG        = '#1c2129'   # page / figure outer background
_AXES      = '#0f1419'   # axes plot area  (darker than page for depth)
_GRID      = '#39424c'   # grid lines / borders
_TEXT      = '#e8e8e8'   # all text
_TICK      = '#b0b8c4'   # axis tick labels (slightly muted)
_BLUE      = '#6CA4F8'   # primary line colour (equity curve, rolling metrics)
_HDR_BG    = '#1a2d50'   # table header background (blue-tinted dark)

# QuantStats hardcodes these colours in its SVG output — map each to our palette
_SVG_SUBS = [
    # axes/figure white backgrounds (attribute form and style form)
    ('fill="#ffffff"',   f'fill="{_AXES}"'),
    ('fill: #ffffff',    f'fill: {_AXES}'),
    # mid-gray tick labels (CSS g[id^="text_"] rule above covers titles/labels too)
    ('fill: #666666',    f'fill: {_TICK}'),
    # Distribution of Monthly Returns — bell curve (black → amber) and bar edges (white → none)
    ('fill: none; stroke: #000000; stroke-width: 1.5; stroke-linecap: round',
     'fill: none; stroke: #f0a050; stroke-width: 1.5; stroke-linecap: round'),
    ('fill: #6CA4F8; fill-opacity: 0.7; stroke: #ffffff',
     'fill: #6CA4F8; fill-opacity: 0.7; stroke: none'),
    # Return Quantiles bar edges and circles — too dark on dark bg
    ('fill: #494949',    'fill: #8090a8'),
    ('stroke: #494949',  'stroke: #8090a8'),
    # Drawdown highlight: 10% opacity is invisible on dark bg — raise to 40% with lighter red
    ('fill: #ff0000; opacity: 0.1; stroke: #ff0000', 'fill: #ff5252; opacity: 0.25; stroke: #ff5252'),
    # QuantStats default strategy-line blue
    ('stroke: #348dc1',  f'stroke: {_BLUE}'),
    ('fill: #348dc1',    f'fill: {_BLUE}'),
    ('fill="#348dc1"',   f'fill="{_BLUE}"'),
    ('stroke="#348dc1"', f'stroke="{_BLUE}"'),
]

@contextmanager
def _dark_mpl():
    """Temporarily set matplotlib rcParams to the dark theme (fallback for any mpl-rendered elements)."""
    overrides = {
        'figure.facecolor': _BG,    'axes.facecolor':    _AXES,
        'axes.edgecolor':   _GRID,  'axes.labelcolor':   _TEXT,
        'text.color':       _TEXT,  'xtick.color':       _TICK,
        'ytick.color':      _TICK,  'grid.color':        _GRID,
        'grid.linestyle':   '--',   'lines.color':       _BLUE,
        'patch.facecolor':  _BLUE,  'savefig.facecolor': _BG,
        'savefig.edgecolor':_BG,
    }
    saved = {k: matplotlib.rcParams[k] for k in overrides}
    matplotlib.rcParams.update(overrides)
    try:
        yield
    finally:
        matplotlib.rcParams.update(saved)

def _inject_dark_css(html_path):
    """Post-process QuantStats HTML: patch page chrome CSS and SVG chart colours."""
    with open(html_path, encoding='utf-8') as f:
        html = f.read()

    # ── Page chrome (CSS literals) ────────────────────────────────────────────
    html = html.replace('background:#fff;color:#000',  f'background:{_BG};color:{_TEXT}')
    html = html.replace('background:#eee',             f'background:{_HDR_BG}')
    html = html.replace('color:grey',                  f'color:{_TEXT}')
    html = html.replace('color:#09c',                  f'color:{_BLUE}')
    html = html.replace('color:#069',                  f'color:{_BLUE}')
    html = html.replace('border-top:1px solid #ccc',   f'border-top:1px solid {_GRID}')

    # ── SVG chart content ─────────────────────────────────────────────────────
    for old, new in _SVG_SUBS:
        html = html.replace(old, new)

    # Distribution of Monthly Returns: inject legend for red mean line + amber bell curve.
    # Anchor on the red mean-line style, which is unique to this SVG.
    _dist_anchor = 'stroke: #f0a050; stroke-width: 1.5; stroke-linecap: round'
    if _dist_anchor in html:
        _dist_legend = (
            '<g font-family="Arial" font-size="10">'
            f'<rect x="330" y="47" width="152" height="56" style="fill:{_BG};stroke:{_GRID};stroke-width:0.8"/>'
            '<line x1="338" y1="66" x2="360" y2="66" style="stroke:#ff0000;stroke-width:1.5;stroke-dasharray:5.55,2.4"/>'
            f'<text x="366" y="70" fill="{_TEXT}">Mean Return</text>'
            '<line x1="338" y1="88" x2="360" y2="88" style="stroke:#f0a050;stroke-width:1.5"/>'
            f'<text x="366" y="92" fill="{_TEXT}">Normal Dist.</text>'
            '</g>'
        )
        _anchor_idx = html.find(_dist_anchor)
        _svg_close  = html.find('</svg>', _anchor_idx)
        html = html[:_svg_close] + _dist_legend + html[_svg_close:]

    # matplotlib renders text as path glyphs inside <g id="text_N"> groups with
    # no explicit fill, so they inherit SVG default (black). This CSS fixes all of them.
    extra_css = f'<style>g[id^="text_"]{{fill:{_TEXT}}}</style>'
    html = html.replace('</head>', extra_css + '\n</head>', 1)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

outputPath = os.path.join(os.path.dirname(__file__), 'results')

def OrderBuy(order, timestamp, cash, quantity, tradeLog, remainingOrders):
    cost = order['limit_price'] * order['qty']
    if cash >= cost:
        cash -= cost
        quantity += order['qty']
        tradeLog.append({ # assumes all order is filled exactly at limit price (conservative, which is good) at the moment, all or nothing. It's good assumption for now.
            'timestamp': timestamp,
            'side': 'buy',
            'qty': order['qty'],
            'price': order['limit_price']
        })
    else:
        remainingOrders.append(order)  # can't afford — keep pending
    return cash, quantity

def OrderSell(order, timestamp, cash, quantity, tradeLog):
    sell_qty = min(order['qty'], quantity)
    if sell_qty > 0:
        cash += order['limit_price'] * sell_qty
        quantity -= sell_qty
        tradeLog.append({
            'timestamp': timestamp,
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

    if 'simple' in strategy.__name__:
        print("running simple version...")
        avgPrices = BARS['avgPrice'].to_numpy()
        all_signals = []
        for i in range(1, len(BARS)):
            all_signals.append(strategy(avgPrices[:i+1], **strategy_kwargs))
    else:
        all_signals = strategy(BARS, **strategy_kwargs)

    low   = BARS['Low'].to_numpy()
    high  = BARS['High'].to_numpy()
    close = BARS['Close'].to_numpy()
    index = BARS.index

    for i in range(1, len(BARS)):
        bar_low, bar_high = low[i], high[i]
        timestamp = index[i]

        # Check pending orders
        remainingOrders = []
        for order in pendingOrders:
            goodToBuy  = order['side'] == 'buy'  and bar_low  <= order['limit_price']
            goodToSell = order['side'] == 'sell' and bar_high >= order['limit_price']
            if    goodToBuy : cash, quantity = OrderBuy(order, timestamp, cash, quantity, tradeLog, remainingOrders)
            elif  goodToSell: cash, quantity = OrderSell(order, timestamp, cash, quantity, tradeLog)
            else: remainingOrders.append(order)
        pendingOrders = remainingOrders

        # Place new order from precomputed signal
        orderInfo = all_signals[i - 1]
        if orderInfo['move'] in ('buy', 'sell'):
            pendingOrders.append({
                'side':        orderInfo['move'],
                'qty':         orderInfo['qty'],
                'limit_price': orderInfo['limit_price'],
            })

        # Record portfolio value
        equityValue.append(cash + quantity * close[i])

    equityCurve = pd.Series(equityValue, index=BARS.index[1:], name='equity')
    return tradeLog, equityCurve


def compute_metrics(tradeLog, equityCurve, initial_cash):
    final_value  = equityCurve.iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100

    rolling_max  = equityCurve.cummax()
    max_drawdown = ((equityCurve - rolling_max) / rolling_max).min() * 100

    bar_returns  = equityCurve.pct_change().dropna()
    sharpe       = (bar_returns.mean() / bar_returns.std() if bar_returns.std() > 0 else 0.0)

    # Win rate: pair buys and sells FIFO into round trips
    buys  = [t['price'] for t in tradeLog if t['side'] == 'buy']
    sells = [t['price'] for t in tradeLog if t['side'] == 'sell']
    pairs = list(zip(buys, sells))
    win_rate = (sum(1 for b, s in pairs if s > b) / len(pairs) * 100) if pairs else 0.0

    return {
        'total_return_pct': round(total_return, 4),
        'max_drawdown_pct': round(max_drawdown, 2),
        'sharpe_ratio':     round(sharpe, 3),
        'win_rate_pct':     round(win_rate, 1),
        'n_trades':         len(tradeLog),
        'final_value':      round(final_value, 2),
    }

def _daily_returns(equityCurve):
    return equityCurve.resample('D').last().pct_change().dropna()

def save_results(BARS, equityCurve, symbol, strategy_name):
    t_start      = pd.Timestamp(BARS.index[0])
    t_end        = pd.Timestamp(BARS.index[-1])
    start_str    = t_start.strftime('%y%m%d_%H%M')
    end_str      = t_end.strftime('%H%M') if t_start.date() == t_end.date() else t_end.strftime('%y%m%d_%H%M')
    timeframe    = f"{start_str}-{end_str}"
    symbol_clean = symbol.replace('/', '-')
    os.makedirs(outputPath, exist_ok=True)


    html_path = os.path.join(outputPath, f"{symbol_clean}_{strategy_name}_{timeframe}.html")
    with _dark_mpl():
        qs.reports.html(_daily_returns(equityCurve), benchmark=None,
                        title=f"{symbol} — {strategy_name} (UTC)", output=html_path)
    _inject_dark_css(html_path)
    print(f"tearsheet saved to {os.path.relpath(html_path)}!")
    return html_path
