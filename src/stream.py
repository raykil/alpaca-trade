import os

from alpaca.data.live.crypto import CryptoDataStream

from src.utils import loadConfig

async def liveStream(msg):
    print("liveStream: ", msg)

def LiveCrypto():
    configPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
    c = loadConfig(configPath, "crypto_paper")
    liveCryptoClient = CryptoDataStream(c['api-key'], c['secret-key'])
    liveCryptoClient.subscribe_bars(liveStream, "BTC/USD")
    liveCryptoClient.run()
