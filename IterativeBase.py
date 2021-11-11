'''
Created on 07.11.2021

@author: usam
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class IterativeBase():

    def __init__(self, symbol, start, end, amount, path=None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.position = 0

        if path is None:
            self.path = "../test_data_/test_data/5mins/AFRM_2M_5mins.csv"
        else: 
            self.path = path
        self.get_data()
    

    def get_data(self):
        raw = pd.read_csv(self.path, parse_dates=["date"]).dropna()
        raw.rename(columns={ "askopen": "Open", "askclose": "Close", "askhigh": "High", "asklow": "Low"}, errors="raise", inplace=True)
        raw.set_index("date",inplace=True)
        raw["price"] = raw["Close"]
        #raw = raw.loc[self.start:self.end]
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        self.data = raw

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
        
    def buy_instrument(self, bar):
        date, price = self.get_values(bar)
       
        units = int(self.current_balance / price)
        self.current_balance -= units * price # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    def sell_instrument(self, bar):
        date, price = self.get_values(bar)
        print("{} |  Selling {} for {}".format(date, self.units, round(price, 5)))

        self.current_balance += self.units * price # increases cash balance by "purchase price"
        self.units = 0
        self.trades += 1
        

    
    def print_current_position_value(self, bar):
        date, price = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        date, price = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, bar):
        date, price = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")