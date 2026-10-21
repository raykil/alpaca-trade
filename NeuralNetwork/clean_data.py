import os, sys, glob, json
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TradingTools import load_bars

nn_dir = os.path.dirname(os.path.abspath(__file__))
PRICES = ['Open', 'High', 'Low', 'Close', 'avgPrice']

def session_grid(start, end):
    with open(os.path.join(os.path.dirname(nn_dir), 'config.json')) as f: keys = json.load(f)['crypto_paper']
    calendar = TradingClient(keys['api-key'], keys['secret-key'], paper=True).get_calendar(GetCalendarRequest(start=start, end=end))
    return pd.DatetimeIndex([t for day in calendar for t in pd.date_range(day.open, day.close, freq='5min', inclusive='left')], name='Timestamp')

def clean_bars(bars, grid):
    bars  = bars.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None).reindex(grid)
    close = bars['Close'].ffill()
    return bars.fillna({**dict.fromkeys(PRICES, close), 'Volume': 0, 'tradeCount': 0}).dropna()

if __name__ == "__main__":
    out_dir = os.path.join(nn_dir, 'data_clean')
    os.makedirs(out_dir, exist_ok=True)
    BARS = {path: load_bars(path) for path in sorted(glob.glob(os.path.join(nn_dir, 'data', '*.csv')))}
    grid = session_grid(min(b.index[0] for b in BARS.values()).date(), max(b.index[-1] for b in BARS.values()).date())
    for path, bars in BARS.items():
        out_path, clean = os.path.join(out_dir, os.path.basename(path)), clean_bars(bars, grid)
        clean.to_csv(out_path)
        print(f"\033[1;32mSaved {len(clean)} bars to {os.path.relpath(out_path, os.path.dirname(nn_dir))}\033[0m")
