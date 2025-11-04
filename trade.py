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
                message = {
                    'Open': msg['o'], 'High': msg['h'], 'Low': msg['l'], 'Close': msg['c'], 'Volume': msg['v'], 
                    'avgPrice': msg['vw']
                }
                yield timestamp, message

def plotBars(BARS, axes):
    if axes is None:
        avgPlot = mpf.make_addplot(BARS['avgPrice'], width=1, color='royalblue')
        axes = mpf.plot(BARS[['Open', 'High', 'Low', 'Close', 'Volume']], type='candle', addplot=avgPlot, returnfig=True)[1]
        return axes
    axes[0].clear()
    avgPlot = mpf.make_addplot(BARS['avgPrice'], width=1, color='royalblue', ax=axes[0])
    mpf.plot(BARS[['Open', 'High', 'Low', 'Close', 'Volume']], type='candle', addplot=avgPlot, ax=axes[0])
    plt.pause(0.001)
    return axes

def appendBars(BARS, timestamp, msg):
    BARS.loc[timestamp] = msg
    BARS = BARS.iloc[-29:]
    print(timestamp, BARS.iloc[-1].to_dict())

async def trade(c, symbols: list[str]):
    async with websockets.connect(c['stream_url']) as websocket:
        await authenticate(websocket, c)
        await subscribe(websocket, symbols)
        BARS = initializeBars()
        plt.ion(); axes=None
        async for timestamp, msg in receiveMessages(websocket):
            appendBars(BARS, timestamp, msg)
            axes = plotBars(BARS, axes)

if __name__ == "__main__":
    parser = ArgumentParser(prog='websocket.py', epilog="jkil@nd.edu")
    parser.add_argument('-m', '--mode'    , default="crypto_paper", type=str , help="Keys in config.json. Options: paper, live, crypto_paper.")
    parser.add_argument('-s', '--strategy', default="momentum"    , type=str , help="Options: momentum only for now")
    parser.add_argument('-t', '--symbols' , default=["BTC/USD"]   , help="Options: momentum only for now")
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    config = loadConfig(f"{scriptPath}/config.json", args.mode)
    asyncio.run(trade(config, args.symbols))