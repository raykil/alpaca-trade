"""
python backtest/backtest.py -s reverse_momentum -t BTC/USD -c 100000 -f backtest/HistoricalData/BTC-USD_500m_20260412_2225.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser

from TradingTools import receiveHistoricalData, initializeBars, save_bars, load_bars
from strategies import strategy_map
from BackTestTools import run_backtest, compute_metrics, plotBacktest, save_results

if __name__ == '__main__':
    parser = ArgumentParser(prog='backtest.py', epilog='jkil@nd.edu')
    parser.add_argument('-s', '--strategy', default='reverse_momentum', type=str, help=f"Options: {', '.join(strategy_map.keys())}")
    parser.add_argument('-t', '--symbol'  , default='BTC/USD'    , type=str)
    parser.add_argument('-d', '--duration', default=500          , type=int, help='Number of historical 1-minute bars to fetch')
    parser.add_argument('-c', '--cash'    , default=100_000.0    , type=float, help='Starting cash')
    parser.add_argument('-f', '--file'    , default=None         , type=str, help='Path to a saved CSV of bars (from fetch_data.py); skips live fetch')
    args = parser.parse_args()

    if args.file:
        BARS = load_bars(args.file)
    else:
        HistoricalData = receiveHistoricalData(args.symbol, duration=args.duration)
        BARS = initializeBars(HistoricalData)

    strategy = strategy_map[args.strategy]
    tradeLog, equityCurve = run_backtest(BARS, strategy, initial_cash=args.cash)
    metrics = compute_metrics(tradeLog, equityCurve, initial_cash=args.cash)


    save_results(BARS, tradeLog, equityCurve, metrics, args.symbol, args.strategy, initial_cash=args.cash)
    # plotBacktest(BARS, tradeLog, equityCurve, metrics, args.symbol, args.strategy)