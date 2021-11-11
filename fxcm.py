import pandas as pd
import numpy as np
import fxcmpy
import time
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class MACDTrader():
    
    def __init__(self, instrument, bar_length, EMA_S, EMA_L, signal_mw, units):
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length) 
        self.tick_data = None
        self.raw_data = None
        self.data = None 
        self.ticks = 0
        self.last_bar = None  
        self.units = units
        self.position = 0
        
        #*****************add strategy-specific attributes here******************
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.signal_mw = signal_mw
        #************************************************************************        
    
    def connect(self):
        self.api = fxcmpy.fxcmpy(config_file= "fxcm.cfg")

    def get_most_recent(self, period = "m5", number = 500):
        while True:  
            time.sleep(5)
            df = self.api.get_candles(self.instrument, number = number, period = period, columns = ["bidclose", "askclose"])
            df[self.instrument] = (df.bidclose + df.askclose) / 2
            df = df[self.instrument].to_frame()
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()) - self.last_bar < self.bar_length:
                break
    
    def get_tick_data(self, data, dataframe):
        
        self.ticks += 1
        print(self.ticks, end = " ")
        
        recent_tick = pd.to_datetime(data["Updated"], unit = "ms")
        
        # if a time longer than the bar_lenght has elapsed between last full bar and the most recent tick
        if recent_tick - self.last_bar > self.bar_length:
            self.tick_data = dataframe.loc[self.last_bar:, ["Bid", "Ask"]]
            self.tick_data[self.instrument] = (self.tick_data.Ask + self.tick_data.Bid)/2
            self.tick_data = self.tick_data[self.instrument].to_frame()
            self.resample_and_join()
            self.define_strategy() 
            self.execute_trades()
            
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                             label="right").last().ffill().iloc[:-1])
        self.last_bar = self.raw_data.index[-1]  
        
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************

        df["EMA_S"] = df[self.instrument].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean()            
        df["EMA_L"] = df[self.instrument].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
        df["MACD"] = df.EMA_S - df.EMA_L            
        df["MACD_Signal"] = df.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
        df["position"] = np.where(df["MACD"] > df["MACD_Signal"], 1, 0)
        df.dropna(inplace = True)
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        
            order_info = {}
            target_profit = 1.2
            stop_loss = 0.85            
            close = self.tick_data[self.instrument].iloc[-1]
            if len(order_info) > 0:
                target_profit_price = order_info["target_profit"]
                stop_loss_price = order_info["stop_loss"]
                buy_price = order_info["buy_price"]
             
                if close > target_profit_price:
                    print(f"SELL: Target Profit hit || bought_price: {buy_price}, target profit: {target_profit_price}")
                    order = self.api.create_market_sell_order(self.instrument, self.units)
                    self.report_trade(order, "GOING SHORT")  # go short with full amount
                    self.position = 0 # short position
                    order_info = {}
                elif close < stop_loss_price:
                    print(f"SELL: Stop Loss hit || bought_price: {buy_price}, stop loss: {stop_loss_price}")
                    order = self.api.create_market_buy_order(self.instrument, self.units)
                    self.report_trade(order, "GOING LONG") # go short with full amount
                    self.position = 0 # short position
                    order_info = {}
                           
            elif self.data["position"].iloc[-1] == 1: # signal to go long
                if self.position == 0:
                    order = self.api.create_market_buy_order(self.instrument, self.units)
                    self.report_trade(order, "GOING LONG")# go long with full amount
                    self.position = 1  # long position
                    
                    stop_loss_price = 0
                    target_profit_price = 0
                    
                    stop_loss_price = close * stop_loss
                    target_profit_price = close * target_profit
                    
                    order_info["buy_price"] = close
                    order_info["target_profit"] = target_profit_price
                    order_info["stop_loss"] = stop_loss_price
                    print(f"Stop loss: {stop_loss_price}")
                    print(f"Target: {target_profit_price}")
                    
            elif self.data["position"].iloc[-1] == 0: # signal to go short
                if self.position == 1:
                    order = self.api.create_market_sell_order(self.instrument, self.units)
                    self.report_trade(order, "GOING SHORT") 
                    self.position = 0 # short position
                    order_info = {}
                    

    def report_trade(self, order, going):
        time = order.get_time()
        units = self.api.get_open_positions().amountK.iloc[-1]
        price = self.api.get_open_positions().open.iloc[-1]
        unreal_pl = self.api.get_open_positions().grossPL.sum()
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | Unreal. P&L = {}".format(time, units, price, unreal_pl))
        print(100 * "-" + "\n")





if __name__ == "__main__":
    instrument = "ETH/USD"
    bar_size = "5min"
    ema_s = 19
    ema_l = 23
    signal = 10
    
    start = time.time()
    while(True): 
        
        trader = MACDTrader(instrument, bar_size, ema_s, ema_l, signal, 1)
        #trader.connect()
        #api.close()
        api = fxcmpy.fxcmpy(config_file= "fxcm.cfg")
        trader.get_most_recent()
        trader.api.subscribe_market_data(instrument, (trader.get_tick_data, ))
        trader.api.unsubscribe_market_data(instrument)
        
        time.sleep(300 - ((time.time() - start) % 300.0))
    