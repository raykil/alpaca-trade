"""
Input : NeuralNetwork/data_npz/{train,valid}.npz, latest NeuralNetwork/{yymmdd}_models/mlp_w{window}.pt
Output: prints always-hold, logistic regression and MLP scores on the same windows, and plots their ROC curves to roc_{split}.png.
"""

import os, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from argparse import ArgumentParser
from train import WindowDataset, build_mlp, run_epoch, device
scriptPath = os.path.dirname(os.path.abspath(__file__))

def inference(model, dataset):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(dataset[batch][0]) for batch in torch.arange(len(dataset), device=device).split(8192)])

def roc(isPositive, scores):
    order    = np.argsort(-scores)
    distinct = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    tpr, fpr = (np.r_[0, np.cumsum(hits[order])[distinct] / hits.sum()] for hits in (isPositive, ~isPositive))
    return fpr, tpr

def score(name, logits, labels, lossFn):
    probs, y = logits.softmax(1).cpu().numpy(), labels.cpu().numpy()
    curves   = [roc(y == k, probs[:, k]) for k in range(3)]
    macroAUC = np.mean([np.trapezoid(tpr, fpr) for fpr, tpr in curves])
    fprGrid  = np.linspace(0, 1, 201)
    plt.plot(fprGrid, np.mean([np.interp(fprGrid, fpr, tpr) for fpr, tpr in curves], axis=0), label=f'{name} (AUC {macroAUC:.3f})')
    print(f"{name:22} loss {lossFn(logits, labels):.4f}  macroAUC {macroAUC:.3f}  acc {(probs.argmax(1) == y).mean():.3f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-w', '--window', default=24, type=int)
    parser.add_argument('-s', '--split', default='valid', help='Split to score. Score test only once, at the very end.')
    parser.add_argument('-e', '--epochs', default=5, type=int, help='Logistic regression epochs.')
    parser.add_argument('-b', '--batch', default=1024, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    args = parser.parse_args()
    torch.manual_seed(0)

    # ————— Load windows & loss ——————————————————————————————————————————————————————
    train, rows = (WindowDataset(os.path.join(scriptPath, 'data_npz', f'{name}.npz'), args.window) for name in ['train', args.split])
    nFeatures = train.features.shape[1]
    counts    = torch.bincount(train.labels, minlength=3)
    lossFn    = nn.CrossEntropyLoss(weight=counts.sum() / (3 * counts.float()))

    # ————— Train logistic regression baseline ———————————————————————————————————————
    logreg    = nn.Sequential(nn.Flatten(), nn.Linear(args.window * nFeatures, 3)).to(device)
    optimizer = torch.optim.AdamW(logreg.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        run_epoch(logreg, train, lossFn, args.batch, optimizer)

    # ————— Load latest MLP ——————————————————————————————————————————————————————————
    mlpPath = sorted(glob.glob(os.path.join(scriptPath, '*_models', f'mlp_w{args.window}.pt')))[-1]
    mlp = build_mlp(args.window, nFeatures)
    mlp.load_state_dict(torch.load(mlpPath, map_location=device)['model'])

    # ————— Score models —————————————————————————————————————————————————————————————
    print(f"Scoring {len(rows):,} {args.split} windows (uniform guess loss = {np.log(3):.4f}, AUC = 0.5)")
    score('always-hold (prior)', torch.log(counts / counts.sum()).expand(len(rows), 3), rows.labels, lossFn)
    score('logistic regression', inference(logreg, rows), rows.labels, lossFn)
    score(f'MLP {os.path.basename(os.path.dirname(mlpPath))}', inference(mlp, rows), rows.labels, lossFn)

    # ————— Plot ROC curves ——————————————————————————————————————————————————————————
    plt.plot([0, 1], [0, 1], 'k:', label='random guess')
    plt.gca().set(xlabel='False positive rate', ylabel='True positive rate', title=f'Macro one-vs-rest ROC ({args.split})')
    plt.legend()
    plt.grid(alpha=0.3)
    rocPath = os.path.join(scriptPath, f'roc_{args.split}.png')
    plt.savefig(rocPath, dpi=300, bbox_inches='tight')
    print(f"\033[1;32mSaved ROC curve to {os.path.relpath(rocPath, os.path.dirname(scriptPath))}\033[0m")
