import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser

from TradingTools import receiveHistoricalData, initializeBars, save_bars, load_bars
from strategies import strategy_map
from BackTestTools import run_backtest, compute_metrics, plotBacktest

if __name__ == '__main__':
    parser = ArgumentParser(prog='backtest.py', epilog='jkil@nd.edu')
    parser.add_argument('-s', '--strategy', default='momentum_v2', type=str, help=f"Options: {', '.join(strategy_map.keys())}")
    parser.add_argument('-t', '--symbol',   default='BTC/USD',     type=str)
    parser.add_argument('-d', '--duration', default=500,           type=int, help='Number of historical 1-minute bars to fetch')
    parser.add_argument('-c', '--cash',     default=100_000.0,     type=float, help='Starting cash')
    parser.add_argument('-f', '--file',     default=None,          type=str, help='Path to a saved CSV of bars (from fetch_data.py); skips live fetch')
    args = parser.parse_args()

    if args.file:
        BARS = load_bars(args.file)
    else:
        HistoricalData = receiveHistoricalData(args.symbol, duration=args.duration)
        BARS           = initializeBars(HistoricalData)
    strategy       = strategy_map[args.strategy]

    tradeLog, equityCurve = run_backtest(BARS, strategy, initial_cash=args.cash)
    metrics                 = compute_metrics(tradeLog, equityCurve, initial_cash=args.cash)

    print(f"\n{'='*44}")
    print(f"  Strategy    : {args.strategy}")
    print(f"  Symbol      : {args.symbol}")
    print(f"  Bars        : {len(BARS)}")
    print(f"  Trades      : {metrics['n_trades']}")
    print(f"  Win rate    : {metrics['win_rate_pct']}%")
    print(f"  Total return: {metrics['total_return_pct']:+.2f}%")
    print(f"  Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Final value : ${metrics['final_value']:,.2f}")
    print(f"{'='*44}\n")

    plotBacktest(BARS, tradeLog, equityCurve, metrics, args.symbol, args.strategy)
