import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TradingTools import *

def save_bars(BARS, filepath):
    BARS.assign(**BARS[['Open', 'High', 'Low', 'Close', 'avgPrice']].map('{:.2f}'.format)).to_csv(filepath)

if __name__ == '__main__':
    parser = ArgumentParser(prog='fetch_data.py', description='Fetch historical bars and save to a CSV for backtesting.', epilog='jkil@nd.edu')
    parser.add_argument('-t', '--symbol'   , required=True   , type=str, help='Symbol to fetch, e.g. BTC/USD')
    parser.add_argument('-i', '--initial'  , required=True   , type=str, help='Start time in UTC: "YYMMDD_HHMM"')
    parser.add_argument('-f', '--final'    , required=True   , type=str, help='End time in UTC: "YYMMDD_HHMM"')
    parser.add_argument('-s', '--size'     , default='15_min', type=str, help='Bar size as "<N>_<unit>", unit: min (1-59), hour (1-23), day/week (1), month (1,2,3,4,6,12). e.g. 15_min, 1_day')
    parser.add_argument('-o', '--outputdir', default=''      , type=str, help='Directory to save the CSV. Default: backtest/HistoricalData')
    args = parser.parse_args()

    initial = datetime.strptime(args.initial, '%y%m%d_%H%M').replace(tzinfo=timezone.utc)
    final   = datetime.strptime(args.final,   '%y%m%d_%H%M').replace(tzinfo=timezone.utc)

    print(f"\nFetching {args.symbol} {args.size} bars...")
    print(f"t_i: {initial.strftime('%y/%m/%d %H:%M')}  ||  t_f: {final.strftime('%y/%m/%d %H:%M')}  (UTC)")
    HistoricalData = receiveHistoricalData(args.symbol, initial, final, scale=args.size)
    BARS = initializeBars(HistoricalData, include_moveDict=False)

    outputdir = args.outputdir or os.path.join(os.path.dirname(__file__), 'HistoricalData')
    os.makedirs(outputdir, exist_ok=True)
    t_start  = BARS.index[0].strftime('%y%m%d_%H%M')
    t_end    = BARS.index[-1].strftime('%y%m%d_%H%M')
    filename = f"{args.symbol.replace('/', '-')}_{t_start}-{t_end}_{args.size.replace('_', '')}.csv"
    filepath = os.path.join(outputdir, filename)

    save_bars(BARS, filepath)
    print(f"\033[1;32mSaved {len(BARS)} bars to {filepath.replace('/Users/raymondkil/alpaca-trade/','')}\033[0m")