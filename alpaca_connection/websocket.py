"""
This script asynchronously receives alpaca quote messages.
Stock trading time: 9:30am - 4pm EST.
Usage: python alpaca_connection/websocket.py
Raymond Kil, September 2025 (jkil@nd.edu)
"""

import yaml, os, websockets, json, asyncio
from argparse import ArgumentParser

def getConfig(configpath):
    with open(configpath, 'r') as y: 
        config = yaml.safe_load(y)['alpaca']
        key_id     = config["key_id"]
        secret_key = config["secret_key"]
        stream_url = config["stream_url"]
    return key_id, secret_key, stream_url

async def receiveAlpacaData(key_id, secret_key, stream_url):
    async with websockets.connect(stream_url) as websocket:
        # Authentication
        auth_msg = {"action": "auth", "key": key_id, "secret": secret_key}
        await websocket.send(json.dumps(auth_msg))
        print("✅ Authentication message successfully sent!")

        # Subscription
        subscribe_msg = {"action": "subscribe", "trades": ["AAPL"], "quotes": ["AAPL"], "bars": ["AAPL"]}
        await websocket.send(json.dumps(subscribe_msg))
        print("✅ Subscription request successfully made!")

        # Loading data from Alpaca
        async for msg in websocket:
            data = json.loads(msg)
            print("Received:", data)

if __name__ == "__main__":
    parser = ArgumentParser(prog='websocket.py', epilog="jkil@nd.edu")
    parser.add_argument('-m', '--mode', default="paper", type=str, help="'paper' for paper trading sandbox, 'live' for live trading.")
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    config = getConfig(f"{scriptPath}/config.yaml")
    asyncio.run(receiveAlpacaData(*config))