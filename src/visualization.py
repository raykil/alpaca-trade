import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from zoneinfo import ZoneInfo

def appendBars(BARS, msg):
    timezone = "America/New_York"
    timestamp = msg.timestamp.astimezone(ZoneInfo(timezone)).replace(microsecond=0, tzinfo=None)
    message = {'Open': msg.open, 'High': msg.high, 'Low': msg.low, 'Close': msg.close, 'Volume': msg.volume, 'tradeCount': msg.trade_count, 'avgPrice': msg.vwap}
    BARS.loc[timestamp, ['Open', 'High', 'Low', 'Close', 'Volume', 'avgPrice']] = message
    print(f"{'-'*78} {BARS.index[-1]} {'-'*78}")

def figureStyle():
    textcolor = 'whitesmoke'
    candle_colors = mpf.make_marketcolors(up='#2d8b30', down='#a50f12', wick='silver', edge='silver', volume='blue')
    candle_style = mpf.make_mpf_style(
        marketcolors=candle_colors, gridstyle='--', gridcolor='#39424c', facecolor='#22272d', figcolor='#1c2129',
        rc={
            'axes.labelcolor': textcolor, 'axes.edgecolor': textcolor, 'axes.labelsize': 14,
            'figure.titlesize': 16, 'figure.titleweight': 'bold',
            'lines.linewidth': 0.6, 'text.color': textcolor, 'xtick.color': textcolor, 'ytick.color': textcolor
        }
    )
    return candle_style

def plotBars(BARS, axes=None, asset_str=None):
    style = figureStyle()
    if axes is None:
        fig = mpf.figure(style=style, figsize=(14, 10))
        fig.suptitle("BTC/USD")
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88)
        gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax2.set_xlabel("Time (EST)")
        axes = [ax1, ax2]
    else:
        ax1, ax2 = axes
        ax1.clear()
        ax2.clear()

    if asset_str:
        fig = ax1.figure
        for txt in fig.texts[1:]:  # preserve suptitle (index 0)
            txt.remove()
        fig.text(0.08, 0.92, asset_str, color='whitesmoke', fontsize=11, ha='left')

    display = BARS.iloc[-60:]
    mpf.plot(display, ax=ax1, volume=ax2, type='candle', style=style)

    order_bars = display[display['order_qty'].notna()]
    unfilled_lines = []
    for ts, row in order_bars.iterrows():
        x      = display.index.get_loc(ts)
        y      = row['order_limit_price']
        is_buy = row['move'] == 'buy'
        ax1.plot(x, y,
                 marker='^' if is_buy else 'v',
                 color='#00e676' if is_buy else '#ff5252',
                 markersize=10, zorder=5, linestyle='None')
        filled = row['order_filled_qty']
        total  = row['order_qty']
        if filled < total:
            bar_top = display.loc[ts, 'High']
            ax1.annotate(f"{filled:g}/{total:g}", (x, bar_top),
                         xytext=(0, 4),
                         textcoords='offset points',
                         rotation=90, ha='center', va='bottom',
                         color='whitesmoke', fontsize=9)

    buy_handle  = mlines.Line2D([], [], marker='^', color='#00e676', linestyle='None', markersize=9, label='Buy')
    sell_handle = mlines.Line2D([], [], marker='v', color='#ff5252', linestyle='None', markersize=9, label='Sell')
    ax1.legend(handles=[buy_handle, sell_handle],
               loc='upper left', framealpha=0.6,
               facecolor='#22272d', edgecolor='#39424c', labelcolor='whitesmoke')

    plt.pause(0.001)
    return axes
