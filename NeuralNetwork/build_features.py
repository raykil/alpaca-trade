import os, sys, glob
import numpy as np
import pandas as pd
from argparse import ArgumentParser
scriptPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(scriptPath))
from TradingTools import load_bars
from BDT.features import evaluate_features, eval_laggedReturn

SPY_LAGS = [1, 5, 10, 30]

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
    
    return pd.DataFrame({**features, **time_encodings(bars.index), **spyReturns}, index=bars.index)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-t', '--symbol', default='*', type=str, help='Symbol to build features for, e.g. CAT. Default: all symbols in data_clean.')
    args = parser.parse_args()

    cleanPath = os.path.join(scriptPath, 'data_clean')
    spy = load_bars(glob.glob(os.path.join(cleanPath, 'SPY_*.csv'))[0])

    outputPath = os.path.join(scriptPath, 'data_features')
    os.makedirs(outputPath, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(cleanPath, f'{args.symbol}_*.csv'))):
        featurePath = os.path.join(outputPath, os.path.basename(path))
        build_features(load_bars(path), spy).to_csv(featurePath, float_format='%.6g')
        print(f"\033[1;32mSaved features to {os.path.relpath(featurePath, os.path.dirname(scriptPath))}\033[0m")
