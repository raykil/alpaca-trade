"""
This script asynchronously receives alpaca quote messages.
Stock trading time: 9:30am - 4pm EST.
Usage: python alpaca_connection/websocket.py
Raymond Kil, September 2025 (jkil@nd.edu)
"""

import yaml, os, websockets, json, asyncio
from datetime import datetime
from collections import deque
from argparse import ArgumentParser
from strategies import *

def getConfig(configpath, mode):
    with open(configpath, 'r') as y:
        return yaml.safe_load(y)[mode]

def parseQuote(msg: dict):
    bp, ap, t = msg.get("bp"), msg.get("ap"), msg.get("t")
    ts = datetime.fromisoformat(t.replace("Z","+00:00")).timestamp()
    mp = 0.5 * (bp + ap)
    return {"timestamp": round(ts, 5), "bidPrice": round(bp, 5), "askPrice": round(ap, 5), "midPrice": round(mp, 5)}

async def receiveAlpacaData(c, strategy_func, strategy_args, PriceHistory):
    async with websockets.connect(c['stream_url']) as websocket:
        # Authentication
        auth_msg = {"action": "auth", "key": c['key_id'], "secret": c['secret_key']}
        await websocket.send(json.dumps(auth_msg))
        print("✅ Authentication message successfully sent!")

        # Subscription
        subscribe_msg = {"action": "subscribe", "trades": ["AAPL"], "quotes": ["AAPL"], "bars": ["AAPL"]}
        await websocket.send(json.dumps(subscribe_msg))
        print("✅ Subscription request successfully made!")

        # Loading data from Alpaca
        async for msg in websocket:
            data = json.loads(msg) # data is a list of dicts
            for d in data:
                if d["T"]=="q": # if a message is a quote...
                    Q = parseQuote(d)
                    PriceHistory.append((Q['midPrice'], Q['timestamp']))
                    # determine buy or sell
                    move = strategy_func(PriceHistory, *strategy_args)
                    print('move', move)

                    # make Payload
                    if move:
                        payload = makePayload(Q, move)
                        print(payload)

                    # TODO: send the payload

if __name__ == "__main__":
    parser = ArgumentParser(prog='websocket.py', epilog="jkil@nd.edu")
    parser.add_argument('-m', '--mode', default="paper", type=str, help="'paper' for paper trading sandbox, 'live' for live trading.")
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    config = getConfig(f"{scriptPath}/config.yaml", args.mode)
    PriceHistory = deque() # each element is a tuple (mp, ts). Rightmost is most recent, leftmost is oldest.
    asyncio.run(receiveAlpacaData(config, STRATEGIES[args.strategy]['func'], STRATEGIES[args.strategy]['args'], PriceHistory))