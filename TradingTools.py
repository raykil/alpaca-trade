import pandas as pd
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone

from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

def receiveHistoricalData(symbol, duration=70, scale='minutes', start=None, end=None):
    client    = CryptoHistoricalDataClient()
    timeframe = TimeFrame.Day if scale == 'days' else TimeFrame.Minute
    if start and end:
        ti, tf = start, end
    else:
        tf = datetime.now(timezone.utc)
        ti = tf - (timedelta(days=duration) if scale == 'days' else timedelta(minutes=duration))
    params = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=ti, end=tf)
    HistoricalData = []
    bars = client.get_crypto_bars(params)[symbol] # list of dicts
    for bar in bars:
        HistoricalData.append({
            'Timestamp' : bar.timestamp.astimezone(timezone.utc).replace(microsecond=0, tzinfo=None),
            'Open'      : bar.open,
            'High'      : bar.high,
            'Low'       : bar.low,
            'Close'     : bar.close,
            'avgPrice'  : bar.vwap,
            'Volume'    : bar.volume, # number of stocks being traded in one bar (minute).
            'tradeCount': bar.trade_count
        })
    return HistoricalData

def readHistoricalData(filepath, max_line=None):
    HistoricalData = []
    with open(filepath, 'r') as f:
        for line in f.readlines()[:max_line]:
            data = {}
            for item in line.split(", "):
                k, v = item.split(": ")
                data[k] = v
            HistoricalData.append(data)
    return HistoricalData

def save_bars(BARS, filepath):
    BARS.assign(**BARS[['Open', 'High', 'Low', 'Close', 'avgPrice']].map('{:.2f}'.format)).to_csv(filepath)

def load_bars(filepath):
    return pd.read_csv(filepath, sep=r'\s*,\s*', engine='python', index_col='Timestamp', parse_dates=True)

def initializeBars(HistoricalData: list = None, include_moveDict: bool = False):
    columns = {
        'Open'      : pd.Series(dtype='float64'),
        'High'      : pd.Series(dtype='float64'),
        'Low'       : pd.Series(dtype='float64'),
        'Close'     : pd.Series(dtype='float64'),
        'avgPrice'  : pd.Series(dtype='float64'),
        'Volume'    : pd.Series(dtype='float64'),
        'tradeCount': pd.Series(dtype='float64'),
    }
    if include_moveDict:
        columns['moveDict'] = pd.Series(dtype='object')  # {'move', 'qty', 'limit_price'} + {'filled_qty', 'order_id'} once submitted
    b = pd.DataFrame(columns, index=pd.DatetimeIndex([], name='Timestamp'))
    if HistoricalData:
        hd = pd.DataFrame(HistoricalData).set_index('Timestamp')
        b = pd.concat([b, hd]).sort_index()
    return b
