import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg') # save-only: no interactive window
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from argparse import ArgumentParser

from src.visualization import figureStyle

def plotEquityCurve(ax, equity):
    ax.plot(equity['timestamp'], equity['value'], color='#58a6ff', linewidth=1)
    ax.set_ylabel('My asset (USD)', color='whitesmoke')
    ax.set_facecolor('#22272d')
    ax.tick_params(colors='whitesmoke')
    ax.grid(True, linestyle='--', color='#39424c', linewidth=0.5)

if __name__ == '__main__':
    parser = ArgumentParser(prog='show_results.py', description='Visualize a saved backtest result JSON.', epilog='jkil@nd.edu')
    parser.add_argument('-f', '--file', required=True, type=str, help='Path to a results JSON file')
    args = parser.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    metrics = data['metrics']
    trades  = data['trades']
    equity  = pd.DataFrame(data['equity'])
    equity['timestamp'] = pd.to_datetime(equity['timestamp'])


    # ———————————————————— Layout ——————————————————————————————
    style = figureStyle()
    title_txt = os.path.splitext(os.path.basename(args.file))[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor='#1c2129', gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(title_txt, color='whitesmoke', fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.12, hspace=0.35)

    plotEquityCurve(ax1, equity)

    # ———————————————————— Trade markers ——————————————————————————————
    eq_indexed = equity.set_index('timestamp')['value']
    for trade in trades:
        ts     = pd.to_datetime(trade['filled'])
        is_buy = trade['side'] == 'buy'
        nearest_ts = eq_indexed.index[eq_indexed.index.get_indexer([ts], method='nearest')[0]]
        ax1.plot(nearest_ts, eq_indexed[nearest_ts], marker='^' if is_buy else 'v', color='#00e676' if is_buy else '#ff5252', markersize=7, zorder=5, linestyle='None')

    buy_handle  = mlines.Line2D([], [], marker='^', color='#00e676', linestyle='None', markersize=7, label='Buy fill')
    sell_handle = mlines.Line2D([], [], marker='v', color='#ff5252', linestyle='None', markersize=7, label='Sell fill')
    ax1.legend(handles=[buy_handle, sell_handle], loc='upper left', framealpha=0.6, facecolor='#22272d', edgecolor='#39424c', labelcolor='whitesmoke')

    # ———————————————————— Trade prices ——————————————————————————————
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df['filled'] = pd.to_datetime(trade_df['filled'])
        buys  = trade_df[trade_df['side'] == 'buy']
        sells = trade_df[trade_df['side'] == 'sell']
        ax2.scatter(buys['filled'],  buys['price'],  marker='^', color='#00e676', s=40, zorder=5, label='Buy')
        ax2.scatter(sells['filled'], sells['price'], marker='v', color='#ff5252', s=40, zorder=5, label='Sell')

    ax2.set_ylabel('Fill Price $', color='whitesmoke')
    ax2.set_facecolor('#22272d')
    ax2.tick_params(colors='whitesmoke')
    ax2.grid(True, linestyle='--', color='#39424c', linewidth=0.5)
    ax2.legend(loc='upper left', framealpha=0.6, facecolor='#22272d', edgecolor='#39424c', labelcolor='whitesmoke')

    for ax in (ax1, ax2):
        ax.spines[:].set_color('#39424c')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    # ———————————————————— Metrics bar ——————————————————————————————
    m = metrics
    metrics_str = (f"Return: {m['total_return_pct']:+.2f}%    "
                   f"Max drawdown: {m['max_drawdown_pct']:.2f}%    "
                   f"Sharpe: {m['sharpe_ratio']:.3f}    "
                   f"Win rate: {m['win_rate_pct']:.1f}%    "
                   f"Trades: {m['n_trades']}    "
                   f"Final: ${m['final_value']:,.2f}")
    fig.text(0.5, 0.02, metrics_str, ha='center', color='whitesmoke', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#22272d', edgecolor='#39424c', alpha=0.9))

    plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    png_path  = os.path.join(plots_dir, os.path.splitext(os.path.basename(args.file))[0] + '.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Plot saved to {os.path.relpath(png_path)}!")
    plt.close(fig)
