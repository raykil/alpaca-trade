"""
Ouput: NeuralNetwork/data_clean_plots/{symbol}.png

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--inputfile', required=True, type=str, help='CSV of cleaned bars, e.g. NeuralNetwork/data_clean/CAT_231002_1125-260930_2335_5min.csv. Need symbol at beginning of filename.')
args = parser.parse_args()

scriptPath = os.path.dirname(os.path.abspath(__file__))
outputPath = os.path.join(scriptPath, 'data_clean_plots')
os.makedirs(outputPath, exist_ok=True)

symbol = os.path.basename(args.inputfile).split('_')[0]
bars = pd.read_csv(args.inputfile, index_col='Timestamp', parse_dates=True)
if 'clean' not in os.path.basename(os.path.dirname(os.path.abspath(args.inputfile))):
    # if considered as not cleaned data, i.e., time zone is in UTC.
    bars.index = bars.index.tz_localize('UTC').tz_convert('America/New_York')
minute = bars.index.hour * 60 + bars.index.minute

# —————————— Plotting ——————————————————————————————
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7), dpi=300)

xBinEdges = np.arange(0, 1441, 5) # 570(09:30), 961(16:00)
trades, volume = bars['tradeCount'], bars['Volume']
boxStyle = dict(ha='right', va='top', fontsize=8, family='monospace', bbox=dict(boxstyle='round,rounding_size=0.1', facecolor='white', alpha=0.8))

# TradeCount (Upper plot)
*_, image1 = ax1.hist2d(minute, trades, bins=(xBinEdges, np.geomspace(max(trades.min(), 1), trades.max(), 100)), norm='log')
fig.colorbar(image1, ax=ax1, label='Bar count')
ax1.set(yscale='log', ylim=[1e1, 1e6], ylabel='TradeCount')
ax1.grid(linestyle='-.')
stats1 = f"min {trades.min():>11,.0f} at {trades.idxmin():%H:%M}\nmax {trades.max():>11,.0f} at {trades.idxmax():%H:%M}"
ax1.text(0.98, 0.95, stats1, transform=ax1.transAxes, **boxStyle)

# Volume (Lower Plot)
*_, image2 = ax2.hist2d(minute, volume, bins=(xBinEdges, np.geomspace(max(volume.min(), 1), volume.max(), 100)), norm='log')
fig.colorbar(image2, ax=ax2, label='Bar count')
ax2.set(yscale='log', ylim=[1e2, 1e8], ylabel='Volume')
ax2.grid(linestyle='-.')
stats2 = f"min {volume.min():>11,.0f} at {volume.idxmin():%H:%M}\nmax {volume.max():>11,.0f} at {volume.idxmax():%H:%M}"
ax2.text(0.98, 0.95, stats2, transform=ax2.transAxes, **boxStyle)

ax1.vlines([570, 960], 0, 1, transform=ax1.get_xaxis_transform(), colors='blue', linestyles='--', label='Regular trading hours')
ax2.vlines([570, 960], 0, 1, transform=ax2.get_xaxis_transform(), colors='blue', linestyles='--')
ax1.legend(loc='upper left', fontsize=8)

ax2.set_xticks(range(0, 1441, 120), [f'{m // 60:02d}:00' for m in range(0, 1441, 120)])
ax1.set_title(f"Trade Counts in Normal Trade Hour ({symbol})")
plt.tight_layout()
plotPath = os.path.join(outputPath, f'{symbol}.png')
plt.savefig(plotPath)
print(f"\033[1;32m{os.path.relpath(plotPath, os.path.dirname(scriptPath))} saved!\033[0m")