import pandas as pd
import numpy as np
import fxcmpy
import time
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use("seaborn")

from strategies.MACDBacktester import MACDBacktester as MACDTester

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
        self.tc = 0.00007
        #*****************add strategy-specific attributes here******************
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.signal_mw = signal_mw
        #************************************************************************        
        self.api = None
    def connect(self):
        self.api = fxcmpy.fxcmpy(config_file= "fxcm.cfg")

    def compute_optimal_parameters(self):
        df = self.raw_data.copy()
        df["Close"] = df[self.instrument]
        macdTester = MACDTester(self.instrument, df, self.EMA_S, self.EMA_L, self.signal_mw, self.tc)
        macdTester.optimize_parameters((5,20,1), (21,50,1), (5,20,1))
        print("MACD Test Results: ",macdTester.test_strategy())
        parameters =  macdTester.get_parameters()
        self.EMA_S = parameters[0]
        self.EMA_L = parameters[1]
        self.signal_mw = parameters[2]
        print("MACD parameters: ", parameters)
        self.EMA_S, self.EMA_L, self.signal_mw = macdTester.get_parameters()
                        
    def get_most_recent(self, period = "m1", number = 5000):
        while True:  
            time.sleep(10)
            df = self.api.get_candles(self.instrument, number = number, period = period, columns = ["bidclose", "askclose"])
            if len(df)==0: 
                print("ERRROR: no data was received on calling get_candles- Method..")
                return -1
            df[self.instrument] = (df.bidclose + df.askclose) / 2
            df = df[self.instrument].to_frame()
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()) - self.last_bar < self.bar_length:
                #self.compute_optimal_parameters()
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
            target_profit = 1.05
            stop_loss = 0.95            
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
    ema_s = 12
    ema_l = 16
    signal = 9
    
    trader = MACDTrader(instrument, bar_size, ema_s, ema_l, signal, 1)
    api = fxcmpy.fxcmpy(config_file= "fxcm.cfg")
    
    trader.get_most_recent()
    api.subscribe_market_data(instrument, (trader.get_tick_data, ))
    
    starttime = time.time()
    timeout = time.time() + 60*60*6
    while time.time() <= timeout:
        time.sleep(900 - ((time.time() - starttime) % 900.0))
        api.unsubscribe_market_data(instrument)
        api.close()

    