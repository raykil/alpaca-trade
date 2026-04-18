import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from datetime import datetime, timezone
from TradingTools import receiveHistoricalData, initializeBars, save_bars

VALID_SCALES = ('minute', 'day')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'HistoricalData')

if __name__ == '__main__':
    parser = ArgumentParser(prog='fetch_data.py', description='Fetch historical bars and save to a CSV for backtesting.', epilog='jkil@nd.edu')
    parser.add_argument('-t', '--symbol'  , required=True   , type=str, help='Symbol to fetch, e.g. BTC/USD')
    parser.add_argument('-d', '--duration', default=500     , type=int, help='Number of bars to fetch (ignored if -i/-f given)')
    parser.add_argument('-s', '--size'    , default='minute', type=str, choices=VALID_SCALES, help='Bar size: minute or day')
    parser.add_argument('-o', '--output'  , default=None    , type=str, help='Output filename (default: auto-generated)')
    parser.add_argument('-i', '--initial' , default=None    , type=str, help='Start time in UTC: "YYYY-MM-DD HH:MM"')
    parser.add_argument('-f', '--final'   , default=None    , type=str, help='End time in UTC: "YYYY-MM-DD HH:MM"')
    args = parser.parse_args()

    if bool(args.initial) != bool(args.final):
        parser.error('-i/--initial and -f/--final must be used together')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FMT     = '%Y-%m-%d %H:%M'
    initial = datetime.strptime(args.initial, FMT).replace(tzinfo=timezone.utc) if args.initial else None
    final   = datetime.strptime(args.final,   FMT).replace(tzinfo=timezone.utc) if args.final   else None

    print(f"Fetching {args.symbol} {args.size} bars{f' from {args.initial} to {args.final} UTC' if initial else f' (last {args.duration})'}...")
    HistoricalData = receiveHistoricalData(args.symbol, duration=args.duration, scale=f'{args.size}s', start=initial, end=final)
    BARS           = initializeBars(HistoricalData)

    if args.output:
        filepath = args.output if os.path.isabs(args.output) else os.path.join(OUTPUT_DIR, args.output)
    else:
        symbol_clean = args.symbol.replace('/', '-')
        timestamp    = datetime.now().strftime('%Y%m%d_%H%M')
        filename     = f"{symbol_clean}_{args.duration}{args.size[0]}_{timestamp}.csv"
        filepath     = os.path.join(OUTPUT_DIR, filename)

    save_bars(BARS, filepath)
    print(f"Saved {len(BARS)} bars to {filepath}")
