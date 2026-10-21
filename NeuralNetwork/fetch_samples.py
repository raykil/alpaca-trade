import os, sys, subprocess

SYMBOLS = [
    # ————— crypto —————
    # 'BTC/USD',  # Bitcoin
    # 'ETH/USD',  # Ethereum
    # 'SOL/USD',  # Solana
    # 'LTC/USD',  # Litecoin
    # 'LINK/USD', # Chainlink

    # ————— ETF ————————
    'SPY',  # S&P 500
    'QQQ',  # Nasdaq 100
    'IWM',  # Russell 2000 (small caps)
    'GLD',  # Gold
    'TLT',  # 20+ year Treasury bonds
    'HYG',  # High-yield corporate bonds
    'SLV',  # Silver
    'USO',  # Crude oil
    'EFA',  # Developed markets ex-US
    'EEM',  # Emerging markets
    'XLU',  # Utilities
    'VNQ',  # Real estate (REITs)

    # ————— stock ——————
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'NVDA',  # Nvidia
    'AMZN',  # Amazon
    'TSLA',  # Tesla
    'JPM',   # JPMorgan Chase
    'XOM',   # ExxonMobil
    'UNH',   # UnitedHealth
    'WMT',   # Walmart
    'KO',    # Coca-Cola
    'GOOGL', # Alphabet
    'META',  # Meta
    'AMD',   # AMD
    'NFLX',  # Netflix
    'BAC',   # Bank of America
    'V',     # Visa
    'LLY',   # Eli Lilly
    'JNJ',   # Johnson & Johnson
    'PG',    # Procter & Gamble
    'COST',  # Costco
    'HD',    # Home Depot
    'MCD',   # McDonald's
    'CAT',   # Caterpillar
    'BA',    # Boeing
    'DIS',   # Disney
]

nn_dir = os.path.dirname(os.path.abspath(__file__))
fetch_script = os.path.join(os.path.dirname(nn_dir), 'backtest', 'fetch_data.py')
for symbol in SYMBOLS:
    subprocess.run([
        sys.executable, fetch_script, 
        '-t', symbol, 
        '-i', '231001_0000', 
        '-f', '260930_2355', 
        '-s', '5_min',
        '-o', os.path.join(nn_dir, 'data')
    ])
