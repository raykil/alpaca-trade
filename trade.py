import os, json, asyncio, websockets
from argparse import ArgumentParser
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
    b = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'avgPrice'])
    b.index = pd.DatetimeIndex([], name='Timestamp')
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

def plotBars(BARS, axes, symbols):
    textcolor = 'whitesmoke'
    candle_colors = mpf.make_marketcolors(up='limegreen', down='red', wick='silver', edge='silver', volume='blue')
    candle_style = mpf.make_mpf_style(marketcolors=candle_colors)
    plotConfig = {
        'avgLine'   : {'width': 1, 'color': 'royalblue', 'label':'Volume-weighted average price'},
        'avgScatter': {'type': 'scatter', 'color': 'royalblue'},
        'candle'    : {'type': 'candle' , 'returnfig': True, 'figsize':(14,8), 'style': candle_style},
        'title'     : {'color': textcolor, 'fontsize': 16, 'fontweight': 'bold'},
        'labels'    : {'color': textcolor, 'fontsize': 14},
        'tickmarks' : {'colors': textcolor}
    }
    if axes is None:
        fig, axes = mpf.plot(BARS[['Open', 'High', 'Low', 'Close', 'Volume']], **plotConfig['candle'])
    else:
        axes[0].clear()
        for p in ['avgLine', 'avgScatter', 'candle']: plotConfig[p]['ax'] = axes[0]
        additional_plots = [
            mpf.make_addplot(BARS['avgPrice'], **plotConfig['avgLine']),
            mpf.make_addplot(BARS['avgPrice'], **plotConfig['avgScatter'])
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
    axes[0].set_xlabel("Time(hh:mm) EST", **plotConfig['labels'])
    axes[0].set_ylabel("Price (USD)", **plotConfig['labels'])
    axes[0].tick_params(axis='x', **plotConfig['tickmarks'])
    axes[0].tick_params(axis='y', **plotConfig['tickmarks'])
    axes[0].grid(which='both', color="#39424c")
    axes[0].set_title(''.join(symbols), **plotConfig['title'])
    axes[0].set_position([L, pos.y0, R-L, pos.height])
    axes[0].margins(x=0.001)
    plt.pause(0.001)
    return axes

def appendBars(BARS, timestamp, msg):
    BARS.loc[timestamp] = msg
    print(timestamp, BARS.iloc[-1].to_dict())

async def trade(c, symbols: list[str]):
    async with websockets.connect(c['stream_url']) as websocket:
        await authenticate(websocket, c)
        await subscribe(websocket, symbols)
        BARS = initializeBars()
        plt.ion(); axes=None
        async for timestamp, msg in receiveMessages(websocket):
            appendBars(BARS, timestamp, msg)
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