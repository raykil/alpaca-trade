"""
Input : NeuralNetwork/data_features/*.csv
Output: NeuralNetwork/data_split/{train,valid,test,holdout}.npz
"""

import os, glob
import numpy as np
import pandas as pd
from argparse import ArgumentParser
scriptPath = os.path.dirname(os.path.abspath(__file__))

notForTraining = ['XOM', 'V']
TIME_COLS = ['tod_sin', 'tod_cos', 'dow_sin', 'dow_cos']

def split_days(index, ratio=(8, 2)):
    """ratio: (train, valid) or (train, valid, test)"""
    days = np.unique(index.date)
    cuts = np.round(np.cumsum(ratio)[:-1] / sum(ratio) * len(days)).astype(int)
    return np.split(days, cuts)

def load_features(path):
    features = pd.read_csv(path, index_col='Timestamp', parse_dates=True).replace([np.inf, -np.inf], np.nan)
    return features.assign(ticker=os.path.basename(path).split('_')[0])

def ZScoreNormalization(rows, avg, std):
    """Z-score normalization: transforms input features to have mean 0 and std 1."""
    clip = 5
    normalized_features = ((rows[avg.index] - avg) / std).clip(-clip, clip)
    return normalized_features

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-r', '--ratio', nargs='+', type=int, default=[8, 2], help='Day split ratio: train valid [test]. Default: 8 2.')
    args = parser.parse_args()

    dataPaths = sorted(glob.glob(os.path.join(scriptPath, 'data_features', '*.csv')))
    data = pd.concat([load_features(path) for path in dataPaths])
    holdout = data['ticker'].isin(notForTraining)
    dates = pd.Index(data.index.date)
    splits = {}
    for name, days in zip(['train', 'valid', 'test'], split_days(data.index, args.ratio)):
        splits[name] = data[dates.isin(days) & ~holdout]
    splits['holdout'] = data[holdout]

    featureCols = data.columns.drop(['label', 'ticker'])
    avg = splits['train'][featureCols].mean() ; avg[TIME_COLS] = 0
    std = splits['train'][featureCols].std()  ; std[TIME_COLS] = 1

    outputPath = os.path.join(scriptPath, 'data_split')
    os.makedirs(outputPath, exist_ok=True)
    for name, rows in splits.items():
        normFeatures = ZScoreNormalization(rows, avg, std)
        splitPath = os.path.join(outputPath, f'{name}.npz')
        np.savez(splitPath,
            normFeatures = normFeatures.to_numpy(np.float32),
            labels       = rows['label'].to_numpy(np.float32), 
            ticker       = rows['ticker'].to_numpy(str),
            timestamp    = rows.index.to_numpy(),
            columns      = featureCols.to_numpy(str), 
            mean         = avg.to_numpy(), 
            std          = std.to_numpy()
        )
        print(f"\033[1;32mSaved {len(rows)} rows to {os.path.relpath(splitPath, os.path.dirname(scriptPath))}\033[0m")
