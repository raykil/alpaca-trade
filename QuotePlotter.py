from time import localtime, strftime
from datetime import datetime, timedelta
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

class QuotePlotter:

    def __init__(self):
        # Initializing data allocation
        self.maxlength = 50
        self.TIMESTAMPS = deque(maxlen=self.maxlength)
        self.BID_PRICES = deque(maxlen=self.maxlength)
        self.ASK_PRICES = deque(maxlen=self.maxlength)
        self.BUY_PRICES = deque(maxlen=self.maxlength)
        self.BUY_TMSTMP = deque(maxlen=self.maxlength)
        self.SEL_PRICES = deque(maxlen=self.maxlength)
        self.SEL_TMSTMP = deque(maxlen=self.maxlength)

        # Initializing plots
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.bidPrice_line = self.ax.plot([], [], label="bidPrice", marker='o')[0]
        self.askPrice_line = self.ax.plot([], [], label="askPrice", marker='o')[0]
        self.buy_marks = self.ax.plot([], [], linestyle="none", marker="^", markersize=9, label="BUY",  zorder=5)[0]
        self.sel_marks = self.ax.plot([], [], linestyle="none", marker="v", markersize=9, label="SELL", zorder=5)[0]

        # Styling plots
        self.recent_plot_time = datetime.fromtimestamp(0)
        self.plot_interval = timedelta(seconds=0.1)
        self.ax.set_title(f"Quotes for APPL")
        self.ax.set_xlabel("Time (hh.mm.ss.sss) EST")
        self.ax.set_ylabel("Price ($)")
        self.ax.set_ylim([254, 256])
        self.ax.grid()
        self.ax.legend()

        # x-axis time formatting
        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: mdates.num2date(x).strftime("%H.%M.%S.%f")[:-3]))
        self.fig.autofmt_xdate()

    def markTrade(self, Q: dict, side: str):
        # Later it should plot prices that were actually bought/sold, not from the payload.
        timestamp = datetime.fromtimestamp(Q['timestamp'])
        # self.TIMESTAMPS.append(timestamp)
        if side == "buy":
            self.BUY_TMSTMP.append(timestamp)
            self.BUY_PRICES.append(Q['askPrice'])
            # self.BUY_PRICES.append(Q['midPrice'])
            self.buy_marks.set_data(self.BUY_TMSTMP, self.BUY_PRICES)
        elif side == "sell":
            self.SEL_TMSTMP.append(timestamp)
            self.SEL_PRICES.append(Q['bidPrice'])
            # self.BUY_PRICES.append(Q['midPrice'])
            self.sel_marks.set_data(self.SEL_TMSTMP, self.SEL_PRICES)
        plt.pause(0.001)


    def update(self, Q: dict):
        timestamp = datetime.fromtimestamp(Q['timestamp'])
        if timestamp - self.recent_plot_time > self.plot_interval:
            bidPrice, askPrice = Q['bidPrice'], Q['askPrice']
            self.TIMESTAMPS.append(timestamp)
            self.BID_PRICES.append(bidPrice)
            self.ASK_PRICES.append(askPrice)
            self.recent_plot_time = timestamp
            self.bidPrice_line.set_data(self.TIMESTAMPS, self.BID_PRICES)
            self.askPrice_line.set_data(self.TIMESTAMPS, self.ASK_PRICES)
            self.ax.relim()
            self.ax.autoscale_view()
            plt.pause(0.001) # This creates the plot window