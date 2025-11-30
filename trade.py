# https://alpaca.markets/sdks/python/market_data.html

import os, json
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from argparse import ArgumentParser
from alpaca.data.live import CryptoDataStream
from strategies import *

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.stream import TradingStream

# CryptoHistoricalDataClient

def loadConfig(configpath, mode):
    with open(configpath, 'r') as c: return json.load(c)[mode]

def initializeBars():
    b = pd.DataFrame({
        'Open':     pd.Series(dtype='float64'),
        'High':     pd.Series(dtype='float64'),
        'Low':      pd.Series(dtype='float64'),
        'Close':    pd.Series(dtype='float64'),
        'Volume':   pd.Series(dtype='float64'),
        'avgPrice': pd.Series(dtype='float64'),
        'move':     pd.Series(dtype='string')
    }, index=pd.DatetimeIndex([], name='Timestamp'))
    return b

def plotBars(BARS, axes, symbols, asset_str, nBars=70):
    BarsToPlot = BARS.copy().iloc[-nBars:]
    textcolor = 'whitesmoke'
    candle_colors = mpf.make_marketcolors(up='#2d8b30', down='#a50f12', wick='silver', edge='silver', volume='blue')
    candle_style = mpf.make_mpf_style(marketcolors=candle_colors)
    plotConfig = {
        'avgLine'    : {'width': 1, 'color': 'royalblue', 'label':'Volume-weighted average price'},
        'avgScatter' : {'type': 'scatter', 'color': 'royalblue', 'markersize': 30},
        'candle'     : {'type': 'candle' , 'returnfig': True, 'figsize':(14,8), 'style': candle_style},
        'buyScatter' : {'type': 'scatter', 'markersize': 180, 'color': "tab:cyan", 'marker': '^', 'label': 'Buy'},
        'sellScatter': {'type': 'scatter', 'markersize': 180, 'color': "tab:orange", 'marker': 'v', 'label': 'Sell'},
        'title'      : {'color': textcolor, 'fontsize': 16, 'fontweight': 'bold'},
        'labels'     : {'color': textcolor, 'fontsize': 14},
        'tickmarks'  : {'colors': textcolor},
        'asset_str'  : {'color': textcolor, 'fontsize': 12, 'ha': 'left'}
    }
    buy_markers  = np.where(BarsToPlot['move']=='buy' , BarsToPlot['avgPrice'], np.nan)
    sell_markers = np.where(BarsToPlot['move']=='sell', BarsToPlot['avgPrice'], np.nan)
    if axes is None:
        fig, axes = mpf.plot(BarsToPlot[['Open', 'High', 'Low', 'Close', 'Volume']], **plotConfig['candle'])
    else:
        axes[0].clear()
        for p in ['avgLine', 'avgScatter', 'candle', 'buyScatter', 'sellScatter']: plotConfig[p]['ax'] = axes[0]
        additional_plots = [
            mpf.make_addplot(BarsToPlot['avgPrice'], **plotConfig['avgLine']),
            mpf.make_addplot(BarsToPlot['avgPrice'], **plotConfig['avgScatter']),
            mpf.make_addplot(buy_markers , **plotConfig['buyScatter']),
            mpf.make_addplot(sell_markers, **plotConfig['sellScatter']),
        ]
        plotConfig['candle'].pop('figsize')
        plotConfig['candle'].pop('returnfig')
        plotConfig['candle']['addplot'] = additional_plots
        mpf.plot(BarsToPlot[['Open', 'High', 'Low', 'Close', 'Volume']], **plotConfig['candle'])
        fig = axes[0].figure
        fig.text(0.02, 0.96, asset_str, **plotConfig['asset_str'])

    # Style
    pos = axes[0].get_position()
    L, R = 0.08, 0.93
    fig.patch.set_facecolor('#1c2129')
    axes[0].set_facecolor('#22272d')
    axes[0].set_xlabel("Time (EST)" , **plotConfig['labels'])
    axes[0].set_ylabel("Price (USD)", **plotConfig['labels'])
    axes[0].tick_params(axis='x', **plotConfig['tickmarks'])
    axes[0].tick_params(axis='y', **plotConfig['tickmarks'])
    axes[0].grid(which='both', color="#39424c")
    axes[0].set_title(''.join(symbols), **plotConfig['title'])
    axes[0].set_position([L, pos.y0, R-L, pos.height])
    axes[0].margins(x=0.001)
    plt.pause(0.001)
    return axes

def appendBars(BARS, msg):
    timestamp = msg.timestamp.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
    message = {'Open': msg.open, 'High': msg.high, 'Low': msg.low, 'Close': msg.close, 'Volume': msg.volume, 'tradeCount': msg.trade_count, 'avgPrice': msg.vwap}
    BARS.loc[timestamp, ['Open', 'High', 'Low', 'Close', 'Volume', 'avgPrice']] = message
    # TODO: instead saving to csv and loading, just call the historical data.
    # BARS.to_csv(f"price_history/{BARS.index[0].strftime("%y%m%dT%H%M")}.csv", index=True)

def makeMove(BARS, strategy, **kwargs):
    # TODO: make it into move dict including all necessary info about building an order.
    move = strategy(BARS, **kwargs)
    return move

def appendMove(BARS, move):
    BARS.loc[BARS.index[-1], 'move'] = move
    print(BARS.iloc[-1].to_dict())

def placeOrder(config, symbol, move, BARS, quantity=0.001):
    if   move=='sell': orderside = OrderSide.SELL
    elif move=='buy' : orderside = OrderSide.BUY
    limitPrice = BARS['avgPrice'].iloc[-1]

    client    = TradingClient(config['api-key'], config['secret-key'], paper=True)
    account   = client.get_account()          # https://alpaca.markets/sdks/python/api_reference/trading/models.html#alpaca.trading.models.TradeAccount
    positions = client.get_all_positions()[0] # https://alpaca.markets/sdks/python/api_reference/trading/models.html#alpaca.trading.models.Position

    order = LimitOrderRequest(
        symbol=symbol, 
        limit_price=limitPrice,
        qty=quantity,
        side=orderside,
        time_in_force=TimeInForce.GTC
    )
    if (float(positions.qty_available)>quantity and move=='sell')|(move=='buy' and float(account.buying_power) >= quantity*limitPrice):
        submitted = client.submit_order(order_data=order)
        print(f"🔵 Order submitted! (ID: {submitted.id})")

def trackOrder(config):
    client = TradingClient(config['api-key'], config['secret-key'], paper=True)
    orders = client.get_orders() # https://alpaca.markets/sdks/python/api_reference/trading/models.html#alpaca.trading.models.Order
    print(f"nOrders: {len(orders)}")
    for order in orders:
        print(
            f"id: {order.id}  "
            f"symbol: {order.symbol}  "
            f"side: {order.side.split('.')[-1]:<4}  "
            f"qty: {order.qty}  "
            f"type: {order.type.split('.')[-1]}  "
            f"status: {order.status.split('.')[-1]:<20}  "
            f"filled_qty: {order.filled_qty:<15}  "
            f"filled_avg_price: {order.filled_avg_price}"
        )

def trackAsset(config):
    client = TradingClient(config['api-key'], config['secret-key'], paper=True)
    account = client.get_account()
    positions = client.get_all_positions()[0]
    asset_str = f"cash:{account.cash}    buying_power:{account.buying_power}    symbol:{positions.symbol}    qty:{positions.qty}    qty_available:{positions.qty_available}"
    return asset_str

def trade(config, symbols, strategy, **strategy_kwargs):
    # This is where historical data logic goes in.
    BARS = initializeBars(); plt.ion(); axes=None
    
    async def recieveMessages(msg):
        nonlocal BARS, axes, config
        appendBars(BARS, msg)
        move = makeMove(BARS, strategy, **strategy_kwargs)
        print(f"{'-'*100} {BARS.index[-1]} {'-'*100}")
        appendMove(BARS, move)
        if move=="buy" or move=="sell": 
            placeOrder(config, symbols[0], move, BARS)
        trackOrder(config)
        asset = trackAsset(config)
        axes = plotBars(BARS, axes, args.symbols, asset)

    return recieveMessages

def receiveData(msg, BARS):
    timestamp = msg.timestamp.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
    message = {'Open': msg.open, 'High': msg.high, 'Low': msg.low, 'Close': msg.close, 'Volume': msg.volume, 'tradeCount': msg.trade_count, 'avgPrice': msg.vwap}
    BARS.loc[timestamp, ['Open','High','Low','Close','Volume','avgPrice']] = message
    # BARS.to_csv(f"price_history/{BARS.index[0].strftime("%y%m%dT%H%M")}.csv", index=True)

if __name__ == "__main__":
    parser = ArgumentParser(prog='websocket.py', epilog="jkil@nd.edu")
    parser.add_argument('-m', '--mode'    , default="crypto_paper", type=str , help="Keys in config.json. Options: paper, live, crypto_paper.")
    parser.add_argument('-s', '--strategy', default="reverse_momentum"    , type=str , help="Options: momentum only for now")
    parser.add_argument('-t', '--symbols' , default=["BTC/USD"])
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    c = loadConfig(f"{scriptPath}/config.json", args.mode)
    client = CryptoDataStream(c['api-key'], c['secret-key'])
    client.subscribe_bars(trade(c, args.symbols, strategy_map[args.strategy]), "BTC/USD")
    client.run()