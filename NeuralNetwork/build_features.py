"""
Input : NeuralNetowrk/data_clean/{symbol}*.csv
Output: .csv files in data_features.
"""

import os, sys, glob
import numpy as np
import pandas as pd
from argparse import ArgumentParser
scriptPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(scriptPath))
from TradingTools import load_bars
from BDT.features import evaluate_features, eval_laggedReturn

SPY_LAGS = [1, 5, 10, 30]
FEE = 0.001

def assign_labels(price, atr, ahead=12, atr_mult=2):
    """
    futureReturn = (price at ahead)/(current price)-1
    If futureReturn > threshold, +1. If futureReturn < -threshold, -1.
    ahead = num of bars ahead to look.
    """
    futurePrice  = price.groupby(price.index.date).shift(-ahead) # price 12 bars ahead
    futureReturn = futurePrice/price - 1
    threshold = np.maximum(FEE, atr_mult*atr)
    label = np.sign(futureReturn).where(futureReturn.abs() > threshold, 0)
    label = label.mask(futureReturn.isna() | atr.isna()) # handling Nans.
    return label

def time_encodings(index):
    session = (index.hour * 60 + index.minute - 570) / 390
    return {'tod_sin': np.sin(2 * np.pi * session),         'tod_cos': np.cos(2 * np.pi * session),
            'dow_sin': np.sin(2 * np.pi * index.dayofweek / 5), 'dow_cos': np.cos(2 * np.pi * index.dayofweek / 5)}

def build_features(bars, spy):
    features = {}
    for name, values in evaluate_features(bars).items():
        if name not in ('bb_bands', 'hour', 'day_of_week'):
            features[name] = values
    
    spyReturns = {}
    for lag in SPY_LAGS:
        spyReturns[f'spy_ret_{lag}'] = eval_laggedReturn(spy['avgPrice'], lag).reindex(bars.index)
    
    label = assign_labels(bars['avgPrice'], features['atr_14'])
    return pd.DataFrame({**features, **time_encodings(bars.index), **spyReturns, 'label': label}, index=bars.index)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--symbol', required=True, type=str, help='Makes csv path: NeuralNetowrk/data_clean/{symbol}*.csv')
    args = parser.parse_args()

    inputPath = os.path.join(scriptPath, 'data_clean')
    spy = load_bars(glob.glob(os.path.join(inputPath, 'SPY_*.csv'))[0])

    outputPath = os.path.join(scriptPath, 'data_features')
    os.makedirs(outputPath, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(inputPath, f'{args.symbol}_*.csv'))):
        featurePath = os.path.join(outputPath, os.path.basename(path))
        build_features(load_bars(path), spy).to_csv(featurePath, float_format='%.6g')
        print(f"\033[1;32mSaved features to {os.path.relpath(featurePath, os.path.dirname(scriptPath))}\033[0m")
