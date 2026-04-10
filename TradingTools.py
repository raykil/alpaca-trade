import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone

from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


def receiveHistoricalData(symbol, duration=70, scale='minutes'):
    client = CryptoHistoricalDataClient()
    tf = datetime.now(timezone.utc)
    if scale == 'days':
        ti = tf - timedelta(days=duration)
        timeframe = TimeFrame.Day
    elif scale == 'minutes':
        ti = tf - timedelta(minutes=duration)
        timeframe = TimeFrame.Minute
    params = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=ti, end=tf)
    HistoricalData = []
    bars = client.get_crypto_bars(params)[symbol]
    for bar in bars:
        HistoricalData.append({
            'Timestamp': bar.timestamp.astimezone(ZoneInfo("America/New_York")).replace(microsecond=0, tzinfo=None),
            'Open':      bar.open,
            'High':      bar.high,
            'Low':       bar.low,
            'Close':     bar.close,
            'Volume':    bar.volume,
            'avgPrice':  bar.vwap,
            'move':      'hold'
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

def initializeBars(HistoricalData: list = None):
    b = pd.DataFrame({
        'Open':              pd.Series(dtype='float64'),
        'High':              pd.Series(dtype='float64'),
        'Low':               pd.Series(dtype='float64'),
        'Close':             pd.Series(dtype='float64'),
        'Volume':            pd.Series(dtype='float64'),
        'avgPrice':          pd.Series(dtype='float64'),
        'move':              pd.Series(dtype='string'),
        'order_qty':         pd.Series(dtype='float64'),
        'order_limit_price': pd.Series(dtype='float64'),
        'order_filled_qty':  pd.Series(dtype='float64'),
        'order_id':          pd.Series(dtype='string'),
    }, index=pd.DatetimeIndex([], name='Timestamp'))
    if HistoricalData:
        hd = pd.DataFrame(HistoricalData).set_index('Timestamp')
        b = pd.concat([b, hd]).sort_index()
    return b
