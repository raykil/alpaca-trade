import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TradingTools import load_bars
from backtest.BackTestTools import PlotStyleDict

def eval_rsi14(price, span=14):
    """
    RSI = Relative Strength Index (measure dPrice/dt) over 14 days.
    """
    delta  = price.diff() # diff wrt bar right before. <class 'pandas.core.series.Series'>
    gain   = delta.clip(lower=0).ewm(alpha=1/span, adjust=False).mean() # ewm = exponentially weighted mass
    loss   = (-delta.clip(upper=0)).ewm(alpha=1/span, adjust=False).mean()
    rsi_14 = 100 - 100 / (1 + gain / loss)
    return rsi_14

def eval_macd(price):
    """
    MACD = Moving Average Convergence Divergence. Difference btw fast/slow EMA (Exponential Moving Average).
    MACD_hist = Acts like momentum-acceleration feature across tickers and price levels.
    """
    ema12 = price.ewm(span=12, adjust=False).mean() # ema = Exponential Moving Average. Moving average with heavier weights in recent data
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd = macd_line / price
    macd_hist = (macd_line - macd_signal) / price
    return macd, macd_hist

def eval_BollingerBands(price, sigma=2):
    """
    bb_pos = 0 when price touches lower band, 1 at upper band.
    """
    bb_mid   = price.rolling(20).mean()
    bb_std   = price.rolling(20).std()
    bb_upper = bb_mid + sigma * bb_std
    bb_lower = bb_mid - sigma * bb_std
    bb_pos   = (price - bb_lower) / (bb_upper - bb_lower)
    bb_width = (bb_upper - bb_lower) / bb_mid
    return bb_pos, bb_width, (bb_upper, bb_lower, bb_mid)

def eval_emasmaCrossover(price, ema_span=10, sma_span=20):
    """
    EMA = Exponential Moving Average (usually faster, more sensitive)
    SMA = Simple Moving Average (usually slower, smoother)
    When EMA goes above SMA, good to buy, when EMA goes below SMA, good to sell.
    """
    ema = price.ewm(span=ema_span, adjust=False).mean()
    sma = price.rolling(sma_span).mean()
    ema_sma_crossover = (ema - sma) / price
    return ema_sma_crossover

def eval_atr(price, high, low, close, span=14):
    prev_close = close.shift(1)
    true_range = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(span).mean() / price
    return atr

def eval_volumeFeatures(price, volume, span=20):
    vol_mean   = volume.rolling(span).mean()
    vol_std    = volume.rolling(span).std().replace(0, np.nan)
    vol_zscore = (volume - vol_mean) / vol_std
    vol_delta  = volume.pct_change()
    roll_vwap  = (price * volume).rolling(20).sum() / volume.rolling(20).sum()
    vwap_dev   = (price - roll_vwap) / roll_vwap
    return vol_zscore, vol_delta, vwap_dev

def eval_laggedReturn(price, lag=1):
    lagged_return = price.pct_change(lag)
    return lagged_return

def evaluate_features(bars):
    price  = bars['avgPrice']
    close  = bars['Close']
    high   = bars['High']
    low    = bars['Low']
    volume = bars['Volume']
    FEATURE_DICT = {}

    FEATURE_DICT['rsi_14']     = eval_rsi14(price)
    FEATURE_DICT['macd']       = eval_macd(price)[0]
    FEATURE_DICT['macd_hist']  = eval_macd(price)[1]
    FEATURE_DICT['bb_pos']     = eval_BollingerBands(price)[0]
    FEATURE_DICT['bb_width']   = eval_BollingerBands(price)[1]
    FEATURE_DICT['bb_bands']   = eval_BollingerBands(price)[2]
    FEATURE_DICT['cross']      = eval_emasmaCrossover(price)
    FEATURE_DICT['atr_14']     = eval_atr(price, high, low, close)
    FEATURE_DICT['vol_zscore'] = eval_volumeFeatures(price, volume)[0]
    FEATURE_DICT['vol_delta']  = eval_volumeFeatures(price, volume)[1]
    FEATURE_DICT['vwap_dev']   = eval_volumeFeatures(price, volume)[2]
    FEATURE_DICT['ret_1']      = eval_laggedReturn(price, lag=1)
    FEATURE_DICT['ret_5']      = eval_laggedReturn(price, lag=5)
    FEATURE_DICT['ret_10']     = eval_laggedReturn(price, lag=10)
    FEATURE_DICT['ret_30']     = eval_laggedReturn(price, lag=30)
    FEATURE_DICT['hour']       = BARS.index.hour
    FEATURE_DICT['day_of_week']= BARS.index.dayofweek

    return FEATURE_DICT

def plot_features(FEATURE_DICT):
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{datetime.now():%y%m%d}_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update(PlotStyleDict)
    time_tag  = os.path.basename(args.inputfile).split('_', 1)[1].rsplit('_', 1)[0]
    timeframe = ' - '.join(datetime.strptime(t, '%y%m%d_%H%M').strftime('%y/%m/%d %H:%M') for t in time_tag.split('-'))

    # —————————— RSI 14 ——————————————————————————————
    ax = FEATURE_DICT['rsi_14'].plot.hist(bins=100, title='RSI 14')
    ax.text(0.02, 0.97, timeframe, transform=ax.transAxes, ha='left', va='top')
    plt.savefig(os.path.join(plot_dir, 'rsi_14.png'), dpi=300, bbox_inches='tight')

    # —————————— Bollinger Bands ——————————————————————————————
    BollingerBands = pd.DataFrame(dict(zip(['upper', 'lower', 'mid'], FEATURE_DICT['bb_bands'])))
    ax = BollingerBands.plot(title='Bollinger Bands', lw=0.8)
    ax.text(0.02, 0.97, timeframe, transform=ax.transAxes, ha='left', va='top')
    plt.savefig(os.path.join(plot_dir, 'bollinger_bands.png'), dpi=300, bbox_inches='tight')

    # —————————— Bollinger postiion/width ——————————————————————————————
    axes = pd.DataFrame({k: FEATURE_DICT[k] for k in ['bb_pos', 'bb_width']}).plot(subplots=True, sharex=True, title='Bollinger Position / Width', lw=0.8)
    axes[0].text(0.02, 0.97, timeframe, transform=axes[0].transAxes, ha='left', va='top')
    plt.savefig(os.path.join(plot_dir, 'bollinger_pos_width.png'), dpi=300, bbox_inches='tight')

    # —————————— Feature distributions ——————————————————————————————
    feats = pd.DataFrame({k: v for k, v in FEATURE_DICT.items() if k not in ('bb_bands', 'hour', 'day_of_week')}).replace([np.inf, -np.inf], np.nan)
    feats[(feats > feats.quantile(0.005)) & (feats < feats.quantile(0.995))].hist(bins=100, layout=(3, 5), figsize=(25, 12))
    plt.suptitle(f'Feature distributions (0.5-99.5 pctl)    {timeframe}')
    plt.savefig(os.path.join(plot_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')

    # —————————— Feature correlation ——————————————————————————————
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(feats.corr('spearman'), cmap='vanimo', vmin=-1, vmax=1)
    ax.set(xticks=range(len(feats.columns)), yticks=range(len(feats.columns)), xticklabels=feats.columns, yticklabels=feats.columns, title=f'Feature correlation (Spearman)\n{timeframe}')
    ax.tick_params(axis='x', rotation=90); ax.grid(False); fig.colorbar(im, ax=ax)
    plt.savefig(os.path.join(plot_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True, type=str, help='CSV of bars, e.g. backtest/HistoricalData/BTC-USD_251018_0000-251021_2358_minute.csv')
    parser.add_argument('-d', '--draw', default=False, action='store_true', help='')
    args = parser.parse_args()

    BARS = load_bars(args.inputfile)
    FEATURE_DICT = evaluate_features(BARS)

    if args.draw:
        plot_features(FEATURE_DICT)

