'''
Created on 07.11.2021

@author: usam
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class IterativeBase():

    def __init__(self, symbol=None, start=None, end=None, units=None, path=None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = 0
        self.current_balance = 0
        self.units = units
        self.trades = 0
        self.position = 0

        if path is None:
            self.path = "test_data/5mins/AFRM_2M_5mins.csv"
        else: 
            self.path = path
        self.get_data()
    
        self.position_info = {}
        self.take_profit_pct = 0.1
        self.stop_loss_pct = 0.05
        
        self.buy_price = 0
        self.sell_price = 0
        
        self.cur_df = None
        self.start_idx = 0
        self.end_idx = 160
        
    def update_current_df(self): 
        
        if len(self.df) > (self.end_idx+1):
            self.cur_df = self.df.iloc[self.start_idx: self.end_idx]
            self.start_idx += 1
            self.end_idx += 1
        else: 
            print("Could not select a subset from the dataframe..")
               
    def init_initial_balance(self, price=None):
        
        if price:
            self.initial_balance = price * self.units
        else: 
            price = self.df["Close"].iloc[(self.end_idx-1)]
            self.initial_balance = round(price * self.units, 2)
            
    def get_data(self):
        raw = pd.read_csv(self.path, parse_dates=["Date"]).dropna()
        #raw.rename(columns={ "askopen": "Open", "askclose": "Close", "askhigh": "High", "asklow": "Low"}, errors="raise", inplace=True)
        raw.set_index("Date",inplace=True)
        raw["price"] = raw["Close"]
        #raw = raw.loc[self.start:self.end]
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        self.df = raw

    def plot_data(self, cols = None):  
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize = (12, 8), title = self.symbol)
    
    def get_values(self, bar):
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        #volume = round(self.data.Volume.iloc[bar], 5)
        return date, price#,volume
    
    def print_current_balance(self, bar):
        date, price = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, price):
        #date, price = self.get_values(bar)
        date = str(self.df.index[self.end_idx])
        self.buy_price = price
        #units = int(self.current_balance / price)
        self.current_balance -= self.units * price # reduce cash balance by "purchase price"
        #self.units += units
        self.trades += 1
        print("{} |  Buying {} for {}".format(date, self.units, round(price*self.units, 5)))
    
    def sell_instrument(self, price):
        date = str(self.df.index[self.end_idx])
        print("{} |  Selling {} for {}".format(date, self.units, round(price*self.units, 5)))

        self.current_balance += self.units * price # increases cash balance by "purchase price"
        #self.units = 0
        self.trades += 1
        self.sell_price = price

    
    def print_current_position_value(self, bar):
        date, price = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        date, price = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, price):
        #date, price = self.get_values(bar)
        date = str(self.df.index[self.end_idx])
        print(75 * "-")
        #print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        print("{} || closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        print(f"{date} || Current Balance: {self.current_balance}")
        print("net performance (%) = {}".format(round(perf, 2)))
        print("number of trades executed = {}".format(self.trades))
        print(75 * "-")
        