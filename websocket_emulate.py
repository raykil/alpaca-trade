"""
This script asynchronously receives alpaca quote messages.
Stock trading time: 9:30am - 4pm EST.
Usage: python alpaca_connection/websocket.py
Raymond Kil, September 2025 (jkil@nd.edu)
"""

import os, asyncio, ast
from collections import deque
from argparse import ArgumentParser
from strategies import *
from websocket import parseQuote

async def receiveAlpacaData(strategy_func, strategy_args, filepath, PriceHistory):
    with open(filepath, 'r') as f:
        for i, msg in enumerate(f):
            print(f"----------msg {i}----------")
            await asyncio.sleep(0.001)
            for d in ast.literal_eval(msg): # equiv to `for d in data:` msg in websocket.py
                if d["T"]=="q": # if a message is a quote...
                    Q = parseQuote(d)
                    PriceHistory.append((Q['midPrice'], Q['timestamp']))
                    # determine buy or sell
                    move = strategy_func(PriceHistory, *strategy_args)
                    print('move', move)

                    # make Payload
                    if move:
                        payload = makePayload(Q, move)
                        # print(payload)

                    # if   move == 'buy' : asyncio.create_task(bracketOrder('buy' , Q["askPrice"]))
                    # elif move == 'sell': asyncio.create_task(bracketOrder('sell', Q["bidPrice"]))

if __name__ == "__main__":
    parser = ArgumentParser(prog='websocket.py', epilog="jkil@nd.edu")
    parser.add_argument('-s', '--strategy', default="momentum", type=str, help="Options: momentum only for now...😅")
    args = parser.parse_args()

    scriptPath = os.path.dirname(os.path.abspath(__file__))
    PriceHistory = deque() # each element is a tuple (mp, ts). Rightmost is most recent, leftmost is oldest.
    asyncio.run(receiveAlpacaData(STRATEGIES[args.strategy]['func'], STRATEGIES[args.strategy]['args'], f"{scriptPath}/appl_clean.log", PriceHistory))