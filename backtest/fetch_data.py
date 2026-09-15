import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TradingTools import *

if __name__ == '__main__':
    parser = ArgumentParser(prog='fetch_data.py', description='Fetch historical bars and save to a CSV for backtesting.', epilog='jkil@nd.edu')
    parser.add_argument('-t', '--symbol'  , required=True   , type=str, help='Symbol to fetch, e.g. BTC/USD')
    parser.add_argument('-d', '--duration', default=720     , type=int, help='Number of bars to fetch (ignored if -i/-f given). Fetches given minutes from present.')
    parser.add_argument('-i', '--initial' , default=None    , type=str, help='Start time in UTC: "YYMMDD_HHMM"')
    parser.add_argument('-f', '--final'   , default=None    , type=str, help='End time in UTC: "YYMMDD_HHMM"')
    parser.add_argument('-s', '--size'    , default='minute', type=str, help='Bar size: minute or day')
    args = parser.parse_args()

    if args.initial and args.final:
        initial = datetime.strptime(args.initial, '%y%m%d_%H%M').replace(tzinfo=timezone.utc)
        final   = datetime.strptime(args.final,   '%y%m%d_%H%M').replace(tzinfo=timezone.utc)
    elif args.duration:
        initial = datetime.now(timezone.utc) - (timedelta(days=args.duration) if args.size == 'day' else timedelta(minutes=args.duration))
        final   = datetime.now(timezone.utc)
    else:
        parser.error('give either -i/--initial with -f/--final, or -d/--duration')

    print(f"\nFetching {args.symbol} {args.size} bars...")
    print(f"t_i: {initial.strftime('%y/%m/%d %H:%M')}  ||  t_f: {final.strftime('%y/%m/%d %H:%M')}  (UTC)")
    HistoricalData = receiveHistoricalData(args.symbol, duration=args.duration, scale=f'{args.size}s', start=initial, end=final)
    BARS = initializeBars(HistoricalData, include_moveDict=False)

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'HistoricalData') # backtest/HistoricalData
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start  = BARS.index[0].strftime('%y%m%d_%H%M')
    t_end    = BARS.index[-1].strftime('%y%m%d_%H%M')
    filename = f"{args.symbol.replace('/', '-')}_{t_start}-{t_end}_{args.size}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    save_bars(BARS, filepath)
    print(f"\033[1;32mSaved {len(BARS)} bars to {filepath.replace('/Users/raymondkil/alpaca-trade/','')}\033[0m")