## How to trade
1. `conda activate alpaca`
2. `trade.py`.

## How to reset Alpaca paper cash
1. Go to [Alpaca dashboard](https://app.alpaca.markets/dashboard/overview)

## How to backtest

**On live data (fetched at runtime):**
1. `conda activate alpaca`
2. `python backtest/backtest.py -s momentum_v2 -t BTC/USD -d 500 -c 100000`

**On fixed saved data:**
1. `conda activate alpaca`
2. Fetch and save bars: `python backtest/fetch_data.py -t BTC/USD -d 500 -i minute`
   - Saved to `backtest/HistoricalData/` with an auto-generated filename
   - Use `-i day` for daily bars, `-d` for number of bars, `-o name.csv` for a custom filename
3. Run backtest on the saved file: `python backtest/backtest.py -f backtest/HistoricalData/<filename>.csv -s momentum_v2`