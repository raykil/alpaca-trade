from BackTestTools import *
import matplotlib.pyplot as plt
scriptPath = os.path.dirname(os.path.abspath(__file__))
rootDir = '/'.join(scriptPath.split('/')[:-1])
sys.path.insert(0, rootDir)
from TradingTools import load_bars
from strategies import strategy_map

def plotEquityCurve(initial_cash, equiCurvePath, plotPath, yScale='percent'):
    equity = pd.read_csv(equiCurvePath, index_col='timestamp', parse_dates=True)['equity']
    if yScale == 'percent':
        equity = (equity / initial_cash - 1) * 100
        ylabel = 'Asset (%)'
    elif yScale == 'cash':
        ylabel = 'Asset (USD)'
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity, color='#58a6ff', linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title("Equity Curve")
    ax.ticklabel_format(axis='y', useOffset=False, style='plain')
    fig.savefig(plotPath, dpi=150, bbox_inches='tight')
    print(f"\033[1;32m{plotPath.replace(rootDir+'/', '')} saved!\033[0m")

if __name__ == '__main__':
    parser = ArgumentParser(prog='backtest.py', epilog='jkil@nd.edu')
    parser.add_argument('-s', '--strategy', default='reverse_momentum', type=str, help=f"Options: {', '.join(strategy_map.keys())}")
    parser.add_argument('-t', '--symbol'  , default='BTC/USD'    , type=str)
    parser.add_argument('-c', '--cash'    , default=100_000.0    , type=float, help='Starting cash')
    parser.add_argument('-f', '--file'    , default=None         , type=str, help='Path to a saved CSV of bars (from fetch_data.py); skips live fetch')
    args = parser.parse_args()

    # ————— Load historical data —————————————————————————————————————————————————————
    BARS = load_bars(args.file)

    # ————— Fetch strategy ———————————————————————————————————————————————————————————
    with open(f"{rootDir}/strategy_params.json") as f: params = json.load(f)
    strategy = strategy_map[args.strategy]
    strategy_kwargs = params.get(args.strategy, {})

    # ————— Output setting —————————————————————————————————————————————————————————————
    scriptPath  = os.path.dirname(os.path.abspath(__file__))
    rootPath = os.path.dirname(scriptPath) # /Users/raymondkil/alpaca-trade
    outputPath = f"{scriptPath}/results/{args.file.split('/')[-1].replace('_minute.csv', '')}"
    os.makedirs(outputPath, exist_ok=True)
    tradeLogsPath = f"{outputPath}/tradeLogs.csv"
    equiCurvePath = f"{outputPath}/equityCurve.csv"
    equiPlotPath  = f"{outputPath}/equityCurve.png"

    # ————— Run backtest —————————————————————————————————————————————————————————————
    tradeLogs, equiCurve = run_backtest(BARS, strategy, initial_cash=args.cash, **strategy_kwargs)
    sharpeRatio = compute_sharpeRatio(equiCurve)
    print(sharpeRatio)
    # metrics = compute_metrics(tradeLogs, equiCurve, initial_cash=args.cash)

    # ————— Save results —————————————————————————————————————————————————————————————
    pd.DataFrame(tradeLogs).to_csv(tradeLogsPath, index=False)
    print(f"\033[1;32m{tradeLogsPath.replace(rootPath+'/', '')} saved!\033[0m")

    equiCurve.to_csv(equiCurvePath, index_label='timestamp', float_format='%.2f')
    print(f"\033[1;32m{equiCurvePath.replace(rootPath+'/', '')} saved!\033[0m")

    # ————— Plot results —————————————————————————————————————————————————————————————
    plt.rcParams.update(PlotStyleDict)
    plotEquityCurve(args.cash, equiCurvePath, equiPlotPath, yScale='cash')

    # html_path = save_results(BARS, equityCurve, args.symbol, args.strategy)

    # # saving json
    # with open(html_path.replace('.html', '.json'), 'w') as f:
    #     json.dump({
    #         'metrics': metrics, 
    #         'trades' : [{**t, 'timestamp': str(t['timestamp'])} for t in tradeLogs], 
    #         'equity' : [{'timestamp': str(ts), 'value': v} for ts, v in equityCurve.items()]
    #     }, f, indent=2)

    # webbrowser.open(f"file://{html_path}")