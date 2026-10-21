import os, json
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone

from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment

def receiveHistoricalData(symbol, start, end, scale='15_min'):
    """scale: ex) 15_min gives 15-minute increment bars (96 bars in a day). Crypto symbols contain '/', e.g. BTC/USD."""
    increment, unit = scale.split('_')
    timeframe = TimeFrame(int(increment), TimeFrameUnit(unit.capitalize()))
    params = {
        'symbol_or_symbols': symbol,
        'timeframe': timeframe,
        'start'    : start,
        'end'      : end
    }

    isCrypto = '/' in symbol
    if isCrypto:
        bars = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(**params))[symbol]
    else:
        configPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        with open(configPath) as f: keys = json.load(f)['crypto_paper']
        bars = StockHistoricalDataClient(keys['api-key'], keys['secret-key']).get_stock_bars(StockBarsRequest(**params, adjustment=Adjustment.ALL))[symbol]
    
    HistoricalData = []
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
