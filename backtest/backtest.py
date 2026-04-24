import os, sys, json
from argparse import ArgumentParser

scriptPath = os.path.dirname(os.path.abspath(__file__))
rootDir = '/'.join(scriptPath.split('/')[:-1])
sys.path.insert(0, rootDir)
from TradingTools import receiveHistoricalData, initializeBars, save_bars, load_bars
from strategies import strategy_map, liveTrade
from BackTestTools import run_backtest, compute_metrics, save_results

if __name__ == '__main__':
    parser = ArgumentParser(prog='backtest.py', epilog='jkil@nd.edu')
    parser.add_argument('-s', '--strategy', default='reverse_momentum', type=str, help=f"Options: {', '.join(strategy_map.keys())}")
    parser.add_argument('-t', '--symbol'  , default='BTC/USD'    , type=str)
    parser.add_argument('-d', '--duration', default=500          , type=int, help='Number of historical 1-minute bars to fetch')
    parser.add_argument('-c', '--cash'    , default=100_000.0    , type=float, help='Starting cash')
    parser.add_argument('-f', '--file'    , default=None         , type=str, help='Path to a saved CSV of bars (from fetch_data.py); skips live fetch')
    args = parser.parse_args()

    # ————— Load historical data —————————————————————————————————————————————————————
    if args.file: BARS = load_bars(args.file)
    else: BARS = initializeBars(receiveHistoricalData(args.symbol, duration=args.duration))

    # ————— Fetch strategy ———————————————————————————————————————————————————————————
    with open(f"{rootDir}/strategy_params.json") as f: params = json.load(f)
    strategy = strategy_map[args.strategy]
    strategy_kwargs = params.get(args.strategy, {})

    # ————— Run backtest —————————————————————————————————————————————————————————————
    tradeLog, equityCurve = run_backtest(BARS, strategy, initial_cash=args.cash, **strategy_kwargs)
    metrics = compute_metrics(tradeLog, equityCurve, initial_cash=args.cash)
    save_results(BARS, equityCurve, args.symbol, args.strategy)