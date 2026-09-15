import os, json
from os.path import join as pathjoin
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PlotStyleDict = { # TODO: later move to a module
    'text.color': '#e8e8e8',
    'font.size': 12,
    'xtick.color': '#b0b8c4',
    'ytick.color': '#b0b8c4',
    'axes.labelcolor': '#e8e8e8',
    'axes.titlecolor': '#e8e8e8',
    'axes.titlesize': 16,
    'axes.edgecolor': '#39424c',
    'axes.facecolor': '#22272d',
    'axes.grid': True,
    'figure.facecolor': '#1c2129',
    'grid.color': '#39424c',
    'grid.linestyle': '--'
}

def binPrices(DATA, granularity=10):
    binnedData = {}
    nRows = len(DATA['Open']) // granularity * granularity
    DATA = {k: v[-nRows:] for k, v in DATA.items()} # cutting to nearest whole number
    for key, values in DATA.items():
        DATA[key] = np.reshape(values, (nRows // granularity, granularity))

    binnedData['Open']   = DATA['Open'][:,0]
    binnedData['Close']  = DATA['Close'][:,-1]
    binnedData['High']   = np.max(DATA['High'], axis=-1)
    binnedData['Low']    = np.min(DATA['Low'], axis=-1)
    binnedData['Volume'] = np.sum(DATA['Volume'], axis=-1)
    binnedData['Timestamp_i'] = DATA['Timestamp'][:,0]
    binnedData['Timestamp_f'] = DATA['Timestamp'][:,-1]
    binnedData['tradeCount']  = np.sum(DATA['tradeCount'], axis=-1)
    binnedData['avgPrice']    = np.where(binnedData['Volume'] > 0, np.sum(DATA['avgPrice'] * DATA['Volume'], axis=-1) / np.maximum(binnedData['Volume'], 1e-12), np.mean(DATA['avgPrice'], axis=-1)) # VWAP, or plain mean if the bin had no volume
    return binnedData

def setRange(DATA, range):
    timestamps = np.array(DATA['Timestamp'], dtype='datetime64[m]')
    t_i, t_f = range.split('-')
    t_i = pd.to_datetime(t_i, format='%y%m%d_%H%M') if len(t_i)>0 else timestamps[0]
    t_f = pd.to_datetime(t_f, format='%y%m%d_%H%M') if len(t_f)>0 else timestamps[-1]
    inRange = (timestamps >= t_i) & (timestamps <= t_f)
    return {k: np.asarray(v)[inRange] for k, v in DATA.items()}

# Ex) python plot_price.py -i HistoricalData/BTC-USD_251018_0000-251021_2358_minute.csv -r 251021_0000-251021_2359 -g 50
parser = ArgumentParser(prog='price_plotter.py', description='Plot a candlestick chart from a historical-bar CSV.', epilog='jkil@nd.edu')
parser.add_argument('-i', '--inputPath'  , required=True, type=str, help='CSV to plot, e.g. HistoricalData/BTC-USD_251018_0000-251021_2358_minute.csv')
parser.add_argument('-g', '--granularity', default=100  , type=int, help='How many source bars are grouped into one candle')
parser.add_argument('-r', '--range'      , default='-'  , type=str, help='Optional: time window to plot. Format: "YYMMDD_HHMM-YYMMDD_HHMM". If one side empty, goes until beginning/end of data.')
args = parser.parse_args()

DATA = pd.read_csv(args.inputPath).to_dict('list')
DATA = setRange(DATA, args.range)
DATA = binPrices(DATA, granularity=args.granularity)

opens   = DATA['Open']
closes  = DATA['Close']
times_i = DATA['Timestamp_i']
times_f = DATA['Timestamp_f']
highs   = DATA['High']
lows    = DATA['Low']
x_axis  = np.arange(len(opens))
bodies  = [abs(ci - oi) for oi, ci in zip(opens, closes)]
bottoms = [min(oi, ci) for oi, ci in zip(opens, closes)] # lower ends of bodies


rangeTag = '-'.join(pd.to_datetime(t).strftime('%y%m%d_%H%M') for t in (times_i[0], times_f[-1]))
symbol = os.path.basename(args.inputPath).split('_')[0]
scriptPath  = os.path.dirname(os.path.abspath(__file__))
rootPath = os.path.dirname(scriptPath) # /Users/raymondkil/alpaca-trade
outputPath  = pathjoin(scriptPath, 'pricePlots') # backtest/pricePlots
outjsonPath = pathjoin(outputPath, 'jsons') ; os.makedirs(outjsonPath, exist_ok=True)
outplotPath = pathjoin(outputPath, 'plots') ; os.makedirs(outplotPath, exist_ok=True)
outjsonFull = pathjoin(outjsonPath, f"{symbol}_{rangeTag}_{args.granularity}min.json")
outplotFull = pathjoin(outplotPath, f"{symbol}_{rangeTag}_{args.granularity}min.png")

# —————————— Save to JSON ————————————————————
BARS_JSON = {str(t): {
    'Open': float(o),
    'High': float(h),
    'Low': float(l),
    'Close': float(c)
    } for t, o, h, l, c in zip(times_i, opens, highs, lows, closes)}

with open(outjsonFull, 'w') as f:
    json.dump(BARS_JSON, f, indent=2)
print(f"\033[1;32mSaved {len(BARS_JSON)} bars to {outjsonFull.replace(rootPath+'/', '')}\033[0m")

# —————————— Plotting ————————————————————
colors = ['#2d8b30' if ci >= oi else '#a50f12' for oi, ci in zip(opens, closes)]
plt.rcParams.update(PlotStyleDict)
fig, ax = plt.subplots(figsize=(14, 7))
for color in set(colors):
    mask = np.array(colors) == color
    ax.errorbar(x_axis[mask], (highs + lows)[mask] / 2, yerr=(highs - lows)[mask] / 2, fmt='none', ecolor=color, elinewidth=0.8, capsize=3) # wisks
ax.bar(x_axis, bodies, bottom=bottoms, width=0.6, color=colors, zorder=2) # bodies
ax.plot(x_axis, DATA['avgPrice'], '.-', color='#6CA4F8', lw=0.8, ms=3, zorder=3, label='avgPrice')

barMinutes = (np.datetime64(times_i[1]) - np.datetime64(times_i[0])) / np.timedelta64(1, 'm')
symbol, timeframe = args.inputPath.split('/')[-1].rsplit('_', 1)[0].split('_', 1)
t_i, t_f = [pd.to_datetime(t).strftime('%y/%m/%d %H:%M') for t in (times_i[0], times_f[-1])]

ax.text(0.02, 0.97, f"1 bar = {barMinutes:g} min", transform=ax.transAxes, ha='left', va='top')
ax.text(0.02, 0.88, f"$t_i$ = {t_i}\n$t_f$ = {t_f}", transform=ax.transAxes, ha='left', va='top', linespacing=2)
ax.set_title(symbol)
ax.legend()
tickStep = int(len(bodies)/12) # denominator ~ nTickmarks
ax.set_xticks(x_axis[::tickStep])
ax.set_xticklabels([t[5:16] for t in times_i[::tickStep]], rotation=45)
ax.set_ylabel("Price (USD)")

fig.savefig(outplotFull, dpi=300, bbox_inches='tight')
print(f"\033[1;32mSaved plot to {outplotFull.replace(rootPath+'/','')}\033[0m")