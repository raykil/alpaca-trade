"""
Input : NeuralNetwork/data/*.csv
Output: NeuralNetwork/data_npz/{train,valid,test,holdout}.npz
"""

import os, sys, glob, json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
scriptPath = os.path.dirname(os.path.abspath(__file__))
rootPath   = os.path.dirname(scriptPath)
sys.path.insert(0, rootPath)
from TradingTools import load_bars
from BDT.features import evaluate_features, eval_laggedReturn

PRICES         = ['Open', 'High', 'Low', 'Close', 'avgPrice']
TIME_COLS      = ['tod_sin', 'tod_cos', 'dow_sin', 'dow_cos']
SPY_LAGS       = [1, 5, 10, 30]
FEE            = 0.001
notForTraining = ['XOM', 'V']

def session_grid(start, end):
    with open(os.path.join(rootPath, 'config.json')) as f: keys = json.load(f)['crypto_paper']
    calendar = TradingClient(keys['api-key'], keys['secret-key'], paper=True).get_calendar(GetCalendarRequest(start=start, end=end))
    return pd.DatetimeIndex([t for day in calendar for t in pd.date_range(day.open, day.close, freq='5min', inclusive='left')], name='Timestamp')

def clean_bars(bars, grid):
    bars  = bars.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None).reindex(grid)
    close = bars['Close'].ffill()
    return bars.fillna({**dict.fromkeys(PRICES, close), 'Volume': 0, 'tradeCount': 0}).dropna()

def assign_labels(price, atr, ahead=12, atr_mult=2):
    """+1/-1 if the same-day return `ahead` bars later beats max(FEE, atr_mult * ATR), else 0."""
    futureReturn = price.groupby(price.index.date).shift(-ahead) / price - 1
    threshold = np.maximum(FEE, atr_mult * atr)
    label = np.sign(futureReturn).where(futureReturn.abs() > threshold, 0)
    return label.mask(futureReturn.isna() | atr.isna())

def time_encodings(index):
    session = (index.hour * 60 + index.minute - 570) / 390
    return {'tod_sin': np.sin(2 * np.pi * session),             'tod_cos': np.cos(2 * np.pi * session),
            'dow_sin': np.sin(2 * np.pi * index.dayofweek / 5), 'dow_cos': np.cos(2 * np.pi * index.dayofweek / 5)}

def build_features(bars, spy):
    features   = {name: values for name, values in evaluate_features(bars).items() if name not in ('bb_bands', 'hour', 'day_of_week')}
    spyReturns = {f'spy_ret_{lag}': eval_laggedReturn(spy['avgPrice'], lag).reindex(bars.index) for lag in SPY_LAGS}
    label      = assign_labels(bars['avgPrice'], features['atr_14'])
    return pd.DataFrame({**features, **time_encodings(bars.index), **spyReturns, 'label': label}, index=bars.index)

def split_days(index, ratio):
    days = np.unique(index.date)
    cuts = np.round(np.cumsum(ratio)[:-1] / sum(ratio) * len(days)).astype(int)
    return np.split(days, cuts)

def ZScoreNormalization(rows, avg, std, clip=5):
    return ((rows[avg.index] - avg) / std).clip(-clip, clip)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-r', '--ratio', nargs='+', type=int, default=[8, 2], help='Day split ratio: train valid [test]. Default: 8 2.')
    args = parser.parse_args()

    # 1. Clean: convert UTC to US/Eastern and reindex every ticker onto the same regular-hours 5-min grid.
    raw   = {os.path.basename(path).split('_')[0]: load_bars(path) for path in sorted(glob.glob(os.path.join(scriptPath, 'data', '*.csv')))}
    grid  = session_grid(min(bars.index[0] for bars in raw.values()).date(), max(bars.index[-1] for bars in raw.values()).date())
    clean = {ticker: clean_bars(bars, grid) for ticker, bars in raw.items()}

    # 2. Features: compute each ticker's features, SPY context and within-day labels, then stack all tickers.
    data = pd.concat([build_features(bars, clean['SPY']).assign(ticker=ticker) for ticker, bars in clean.items()]).replace([np.inf, -np.inf], np.nan)

    # 3. Split: cut by calendar day into train/valid(/test), and keep holdout tickers out of all of them.
    holdout = data['ticker'].isin(notForTraining)
    dates   = pd.Index(data.index.date)
    splits  = {name: data[dates.isin(days) & ~holdout] for name, days in zip(['train', 'valid', 'test'], split_days(data.index, args.ratio))}
    splits['holdout'] = data[holdout]

    # 4. Normalize & save: z-score every split with train-only stats and write one .npz per split.
    featureCols = data.columns.drop(['label', 'ticker'])
    avg = splits['train'][featureCols].mean() ; avg[TIME_COLS] = 0
    std = splits['train'][featureCols].std()  ; std[TIME_COLS] = 1
    outputPath = os.path.join(scriptPath, 'data_npz')
    os.makedirs(outputPath, exist_ok=True)
    for name, rows in splits.items():
        splitPath = os.path.join(outputPath, f'{name}.npz')
        np.savez(splitPath,
            normFeatures = ZScoreNormalization(rows, avg, std).to_numpy(np.float32),
            labels       = rows['label'].to_numpy(np.float32),
            ticker       = rows['ticker'].to_numpy(str),
            timestamp    = rows.index.to_numpy(),
            columns      = featureCols.to_numpy(str),
            mean         = avg.to_numpy(),
            std          = std.to_numpy()
        )
        print(f"\033[1;32mSaved {len(rows)} rows to {os.path.relpath(splitPath, rootPath)}\033[0m")
