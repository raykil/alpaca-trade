import os, json, asyncio, websockets
from argparse import ArgumentParser

def loadConfig(configpath, mode):
    with open(configpath, 'r') as c: return json.load(c)[mode]
    
async def authenticate(websocket, c):
    auth_msg = {"action": 'auth', "key": c['key_id'], "secret": c['secret_key']}
    await websocket.send(json.dumps(auth_msg))
    print("✅ Authentication message successfully sent!")

async def subscribe(websocket, tickers: list[str]): # Ex) tickers: ["AAPL", "TSLA"]
    subscribe_msg = {"action": 'subscribe', "trades": tickers, "quotes": tickers, "bars": tickers}
    await websocket.send(json.dumps(subscribe_msg))
    print("✅ Subscription request successfully made!")

async def receiveQuotes(websocket):
    async for msg in websocket:
        print(f"Original message was: {msg}")
        for quote in json.loads(msg):
            if quote['T']=='q': yield quote

async def trade(c, tickers: list[str]):
    async with websockets.connect(c['stream_url']) as websocket:
        await authenticate(websocket, c)
        await subscribe(websocket, tickers)
        quote_iterator = receiveQuotes(websocket)
        async for quote in quote_iterator:
            print(quote, type(quote))


if __name__ == "__main__":
    parser = ArgumentParser(prog='websocket.py', epilog="jkil@nd.edu")
    parser.add_argument('-m', '--mode', default="paper", type=str, help="Keys in config.json.")
    parser.add_argument('-s', '--strategy', default="momentum", type=str, help="Options: momentum only for now")
    parser.add_argument('-t', '--tickers', default="BTC/USD", type=str, help="Options: momentum only for now")
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    config = loadConfig(f"{scriptPath}/config.json", args.mode)
    asyncio.run(trade(config, ["BTC/USD"]))