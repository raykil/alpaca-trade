import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from alpaca.data.live.crypto import CryptoDataStream

from src.utils import loadConfig
from TradingTools import receiveHistoricalData, initializeBars
from src.visualization import appendBars, plotBars
from src.execution import makeMove, appendMove, placeOrder, trackOrder, trackAsset
from strategies import strategy_map

def trade(config, symbol, strategy, **strategy_kwargs):
    HistoricalData = receiveHistoricalData(symbol, duration=60)
    BARS = initializeBars(HistoricalData)
    plt.ion()
    axes = None

    async def recieveMessages(msg):
        nonlocal axes
        appendBars(BARS, msg)
        order_info = makeMove(BARS, strategy, **strategy_kwargs)
        appendMove(BARS, order_info['move'])
        if order_info['move'] in ('buy', 'sell'):
            order_id = placeOrder(config, symbol, order_info)
            if order_id:
                BARS.loc[BARS.index[-1], 'order_qty']         = order_info['qty']
                BARS.loc[BARS.index[-1], 'order_limit_price'] = order_info['limit_price']
                BARS.loc[BARS.index[-1], 'order_filled_qty']  = 0.0
                BARS.loc[BARS.index[-1], 'order_id']          = order_id
        trackOrder(config, BARS)
        asset_str = trackAsset(config)
        axes = plotBars(BARS, axes, asset_str)

    return recieveMessages

if __name__ == "__main__":
    parser = ArgumentParser(prog='trade.py', epilog="jkil@nd.edu")
    parser.add_argument('-m', '--mode'    , default="crypto_paper"    , type=str, help="Keys in config.json. Options: paper, live, crypto_paper.")
    parser.add_argument('-s', '--strategy', default="reverse_momentum", type=str, help="Options: momentum, reverse_momentum, order_test")
    parser.add_argument('-t', '--symbol'  , default="BTC/USD")
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    c = loadConfig(f"{scriptPath}/config.json", args.mode)
    client = CryptoDataStream(c['api-key'], c['secret-key'])
    client.subscribe_bars(trade(c, args.symbol, strategy_map[args.strategy]), args.symbol)
    client.run()
