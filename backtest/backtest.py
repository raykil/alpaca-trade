import os, sys, json, webbrowser
from argparse import ArgumentParser

scriptPath = os.path.dirname(os.path.abspath(__file__))
rootDir = '/'.join(scriptPath.split('/')[:-1])
sys.path.insert(0, rootDir)
from TradingTools import receiveHistoricalData, initializeBars, load_bars
from strategies import strategy_map
from BackTestTools import run_backtest, compute_metrics, save_results

if __name__ == '__main__':
    parser = ArgumentParser(prog='backtest.py', epilog='jkil@nd.edu')
    parser.add_argument('-s', '--strategy', default='reverse_momentum', type=str, help=f"Options: {', '.join(strategy_map.keys())}")
    parser.add_argument('-t', '--symbol'  , default='BTC/USD'    , type=str)
    # parser.add_argument('-d', '--duration', default=500          , type=int, help='Number of historical 1-minute bars to fetch')
    parser.add_argument('-c', '--cash'    , default=100_000.0    , type=float, help='Starting cash')
    parser.add_argument('-f', '--file'    , default=None         , type=str, help='Path to a saved CSV of bars (from fetch_data.py); skips live fetch')
    args = parser.parse_args()

    # ————— Load historical data —————————————————————————————————————————————————————
    # if args.file: BARS = load_bars(args.file)
    # else: BARS = initializeBars(receiveHistoricalData(args.symbol, duration=args.duration))
    BARS = load_bars(args.file)

    # ————— Fetch strategy ———————————————————————————————————————————————————————————
    with open(f"{rootDir}/strategy_params.json") as f: params = json.load(f)
    strategy = strategy_map[args.strategy]
    strategy_kwargs = params.get(args.strategy, {})

    # ————— Run backtest —————————————————————————————————————————————————————————————
    tradeLogs, equityCurve = run_backtest(BARS, strategy, initial_cash=args.cash, **strategy_kwargs)

    scriptPath  = os.path.dirname(os.path.abspath(__file__))
    rootPath = os.path.dirname(scriptPath) # /Users/raymondkil/alpaca-trade
    outputPath = f"{scriptPath}/results/{args.file.split('/')[-1].replace('_minute.csv', '')}"
    os.makedirs(outputPath, exist_ok=True)

    tradeLogFull = f"{outputPath}/tradeLogs.txt"
    equityCurveFull = f"{outputPath}/equityCurve.txt"

    with open(tradeLogFull, 'w') as f:
        for tradeLog in tradeLogs: f.write(f"{tradeLog}\n")
    print(f"\033[1;32m{tradeLogFull.replace(rootPath+'/', '')} saved!\033[0m")

    with open(equityCurveFull, 'w') as f:
        for timestamp, value in equityCurve.items(): f.write(f"{timestamp}  {value:.2f}\n")
    print(f"\033[1;32m{equityCurveFull.replace(rootPath+'/', '')} saved!\033[0m")

    sys.exit()

    metrics = compute_metrics(tradeLogs, equityCurve, initial_cash=args.cash)


    # html_path = save_results(BARS, equityCurve, args.symbol, args.strategy)

    # # saving json
    # with open(html_path.replace('.html', '.json'), 'w') as f:
    #     json.dump({
    #         'metrics': metrics, 
    #         'trades' : [{**t, 'timestamp': str(t['timestamp'])} for t in tradeLogs], 
    #         'equity' : [{'timestamp': str(ts), 'value': v} for ts, v in equityCurve.items()]
    #     }, f, indent=2)

    # webbrowser.open(f"file://{html_path}")