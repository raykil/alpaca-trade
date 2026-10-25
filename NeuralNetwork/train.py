"""
Input : NeuralNetwork/data_split/{train,valid}.npz
Output: NeuralNetwork/models/mlp_w{window}.pt
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from argparse import ArgumentParser
scriptPath = os.path.dirname(os.path.abspath(__file__))
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class WindowDataset(Dataset):
    """Sample = the last `window` bars of one ticker's session, ending at a labeled bar."""
    def __init__(self, path, window):
        data = np.load(path)
        timestamps = pd.DatetimeIndex(data['timestamp'])
        barOfDay = (timestamps.hour * 60 + timestamps.minute - 570) // 5
        ends = np.flatnonzero((barOfDay >= window - 1) & ~np.isnan(data['labels']))
        self.features = torch.from_numpy(np.nan_to_num(data['normFeatures'])).to(device)
        self.labels = torch.from_numpy(data['labels'][ends] + 1).long().to(device)
        self.ends = torch.from_numpy(ends).to(device)
        self.offsets = torch.arange(1 - window, 1, device=device)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        return self.features[self.ends[i].unsqueeze(-1) + self.offsets], self.labels[i]

def build_mlp(window, nFeatures, hidden=128, dropout=0.2):
    return nn.Sequential(nn.Flatten(), nn.Linear(window * nFeatures, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3)).to(device)

def run_epoch(model, dataset, lossFn, batchSize, optimizer=None):
    model.train(optimizer is not None)
    totalLoss = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in torch.randperm(len(dataset), device=device).split(batchSize):
            windows, labels = dataset[batch]
            loss = lossFn(model(windows), labels)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            totalLoss += loss.detach() * len(batch)
    return totalLoss.item() / len(dataset)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-w', '--window', default=24, type=int, help='Past bars per sample. Default: 24 (2 hours).')
    parser.add_argument('-e', '--epochs', default=30, type=int)
    parser.add_argument('-b', '--batch', default=1024, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--patience', default=3, type=int, help='Epochs without valid improvement before stopping.')
    parser.add_argument('--overfit', default=0, type=int, help='Sanity check: train and validate on the first N windows only.')
    args = parser.parse_args()
    torch.manual_seed(0)

    splitPath = os.path.join(scriptPath, 'data_split')
    train, valid = (WindowDataset(os.path.join(splitPath, f'{name}.npz'), args.window) for name in ['train', 'valid'])
    if args.overfit:
        train.ends, train.labels = train.ends[:args.overfit], train.labels[:args.overfit]
        valid = train

    counts = torch.bincount(train.labels, minlength=3)
    lossFn = nn.CrossEntropyLoss(weight=counts.sum() / (3 * counts.float()))
    model = build_mlp(args.window, train.features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    modelPath = os.path.join(scriptPath, 'models', 'overfit.pt' if args.overfit else f'mlp_w{args.window}.pt')
    os.makedirs(os.path.dirname(modelPath), exist_ok=True)
    bestLoss, badEpochs = np.inf, 0
    for epoch in range(args.epochs):
        trainLoss = run_epoch(model, train, lossFn, args.batch, optimizer)
        validLoss = run_epoch(model, valid, lossFn, args.batch)
        print(f"Epoch {epoch:3}  train {trainLoss:.4f}  valid {validLoss:.4f}")
        if validLoss < bestLoss:
            bestLoss, badEpochs = validLoss, 0
            torch.save({'model': model.state_dict(), 'args': vars(args)}, modelPath)
        elif (badEpochs := badEpochs + 1) == args.patience:
            break
    print(f"\033[1;32mSaved best model (valid loss {bestLoss:.4f}) to {os.path.relpath(modelPath, os.path.dirname(scriptPath))}\033[0m")
