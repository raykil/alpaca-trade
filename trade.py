import os, json, asyncio, websockets
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import mplfinance as mpf

def loadConfig(configpath, mode):
    with open(configpath, 'r') as c: return json.load(c)[mode]
    
async def authenticate(websocket, c):
    auth_msg = {"action": 'auth', "key": c['key_id'], "secret": c['secret_key']}
    await websocket.send(json.dumps(auth_msg))

async def subscribe(websocket, symbols: list[str]):
    subscribe_msg = {"action": 'subscribe', "bars": symbols} # "updatedBars": symbols
    await websocket.send(json.dumps(subscribe_msg))

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

async def receiveMessages(websocket):
    async for messages in websocket:
        for msg in json.loads(messages):
            if   msg['T']=='success' and msg['msg']=='connected'    : print("✅ Connection successful!")
            elif msg['T']=='success' and msg['msg']=='authenticated': print("✅ Authentication successful!")
            elif msg['T']=='subscription': print("✅ Subscribed to:", {k: v for k, v in msg.items() if isinstance(v, list) and v})
            elif (msg['T']=='b')|(msg['T']=='u'):
                timestamp = datetime.fromisoformat(msg['t']).astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
                message = {'Open': msg['o'], 'High': msg['h'], 'Low': msg['l'], 'Close': msg['c'], 'Volume': msg['v'], 'avgPrice': msg['vw']}
                yield timestamp, message

def appendBars(BARS, timestamp, msg, window=70):
    BARS.loc[timestamp, ['Open','High','Low','Close','Volume','avgPrice']] = msg
    BARS = BARS[-window:] # window: Max number of bars to show.

def plotBars(BARS, axes, symbols):
    textcolor = 'whitesmoke'
    candle_colors = mpf.make_marketcolors(up='#2d8b30', down='#a50f12', wick='silver', edge='silver', volume='blue')
    candle_style = mpf.make_mpf_style(marketcolors=candle_colors)
    plotConfig = {
        'avgLine'    : {'width': 1, 'color': 'royalblue', 'label':'Volume-weighted average price'},
        'avgScatter' : {'type': 'scatter', 'color': 'royalblue', 'markersize': 30},
        'candle'     : {'type': 'candle' , 'returnfig': True, 'figsize':(14,8), 'style': candle_style},
        'buyScatter' : {'type': 'scatter', 'markersize': 180, 'color': "tab:blue", 'marker': 'H', 'label': 'Buy'},
        'sellScatter': {'type': 'scatter', 'markersize': 180, 'color': "tab:orange", 'marker': 'v', 'label': 'Sell'},
        'title'      : {'color': textcolor, 'fontsize': 16, 'fontweight': 'bold'},
        'labels'     : {'color': textcolor, 'fontsize': 14},
        'tickmarks'  : {'colors': textcolor}
    }
    buy_markers  = np.where(BARS['move']=='buy' , BARS['avgPrice'], np.nan)
    sell_markers = np.where(BARS['move']=='sell', BARS['avgPrice'], np.nan)
    if axes is None:
        fig, axes = mpf.plot(BARS[['Open', 'High', 'Low', 'Close', 'Volume']], **plotConfig['candle'])
    else:
        axes[0].clear()
        for p in ['avgLine', 'avgScatter', 'candle', 'buyScatter', 'sellScatter']: plotConfig[p]['ax'] = axes[0]
        additional_plots = [
            mpf.make_addplot(BARS['avgPrice'], **plotConfig['avgLine']),
            mpf.make_addplot(BARS['avgPrice'], **plotConfig['avgScatter']),
            mpf.make_addplot(buy_markers , **plotConfig['buyScatter']),
            mpf.make_addplot(sell_markers, **plotConfig['sellScatter']),
        ]
        plotConfig['candle'].pop('figsize')
        plotConfig['candle'].pop('returnfig')
        plotConfig['candle']['addplot'] = additional_plots
        mpf.plot(BARS[['Open', 'High', 'Low', 'Close', 'Volume']], **plotConfig['candle'])
        fig = axes[0].figure

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


def appendMove(BARS, window=3):
    avgPrices = BARS['avgPrice'].to_numpy()[-window:]
    p_i, p_f = avgPrices[0], avgPrices[-1]
    delta_p = p_f-p_i
    if   (len(avgPrices)>=window) & (delta_p> 100): move = 'buy'
    elif (len(avgPrices)>=window) & (delta_p<-100): move = 'sell'
    else: move = 'hold'
    BARS.loc[BARS.index[-1], 'move'] = move
    print(BARS.iloc[-1].to_dict())
    return move


async def trade(c, symbols: list[str]):
    async with websockets.connect(c['stream_url']) as websocket:
        await authenticate(websocket, c)
        await subscribe(websocket, symbols)
        BARS = initializeBars()
        plt.ion(); axes=None
        async for timestamp, msg in receiveMessages(websocket):
            appendBars(BARS, timestamp, msg)
            appendMove(BARS)
            axes = plotBars(BARS, axes, symbols)

if __name__ == "__main__":
    parser = ArgumentParser(prog='websocket.py', epilog="jkil@nd.edu")
    parser.add_argument('-m', '--mode'    , default="crypto_paper", type=str , help="Keys in config.json. Options: paper, live, crypto_paper.")
    parser.add_argument('-s', '--strategy', default="momentum"    , type=str , help="Options: momentum only for now")
    parser.add_argument('-t', '--symbols' , default=["BTC/USD"]   , help="Options: momentum only for now")
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    config = loadConfig(f"{scriptPath}/config.json", args.mode)
    asyncio.run(trade(config, args.symbols))