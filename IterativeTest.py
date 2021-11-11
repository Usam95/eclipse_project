'''
Created on 07.11.2021

@author: usam
'''

from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from IterativeBase import IterativeBase


class IterativeBacktest(IterativeBase):

    # helper method
    def go_long(self, bar): 
        self.buy_instrument(bar) # go long

    # helper method
    def go_short(self, bar, ): 

        self.sell_instrument(bar) # go short
        
    def test_macd_strategy_bracket(self, EMA_S, EMA_L, signal_mw):
        
        # nice printout
        stm = "Testing MACD strategy | {} | EMA_S = {} & EMA_S = {}".format(self.symbol, EMA_S, EMA_S)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        
        if EMA_S is not None:
            self.EMA_S = EMA_S
            self.data["EMA_S"] = self.data["Close"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean()            
        if EMA_L is not None:
            self.EMA_L = EMA_L
            self.data["EMA_L"] = self.data["Close"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
            self.data["MACD"] = self.data.EMA_S - self.data.EMA_L            
        if signal_mw is not None:
            self.signal_mw = signal_mw
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
        self.data.dropna(inplace = True)
    
        order_info = {}
        target_profit = 1.2
        stop_loss = 0.85
        # sma crossover strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            date, price = self.get_values(bar)
            
            if len(order_info) > 0:
                target_profit_price = order_info["target_profit"]
                stop_loss_price = order_info["stop_loss"]
                buy_price = order_info["buy_price"]
             
                if price > target_profit_price:
                    print(f"SELL: Target Profit hit || bought_price: {buy_price}, target profit: {target_profit_price}")
                    self.go_short(bar) # go short with full amount
                    self.position = 0 # short position
                    order_info = {}
                    continue
                elif price < stop_loss_price:
                    self.go_short(bar) # go short with full amount
                    self.position = 0 # short position
                    print(f"SELL: Stop Loss hit || bought_price: {buy_price}, stop loss: {stop_loss_price}")
                    order_info = {}
                    continue
                
            if self.data["MACD"].iloc[bar] > self.data["MACD_Signal"].iloc[bar]: # signal to go long
                if self.position == 0:
                    self.go_long(bar) # go long with full amount
                    self.position = 1  # long position
                    
                    stop_loss_price = 0
                    target_profit_profit = 0
                    
                    stop_loss_price = price * stop_loss
                    target_profit_price = price * target_profit
                    
                    order_info["buy_price"] = price
                    order_info["target_profit"] = target_profit_price
                    order_info["stop_loss"] = stop_loss_price
                    print(f"Stop loss: {stop_loss_price}")
                    print(f"Target: {target_profit_price}")
                    
            elif self.data["MACD"].iloc[bar] < self.data["MACD_Signal"].iloc[bar]: # signal to go short
                if self.position == 1:
                    self.go_short(bar) # go short with full amount
                    self.position = 0 # short position
                    order_info = {}
        self.close_pos(bar+1) # close position at the last bar           
    def test_sma_strategy_bracket(self, SMA_S, SMA_L):
        
        # nice printout
        stm = "Testing SMA strategy | {} | SMA_S = {} & SMA_L = {}".format(self.symbol, SMA_S, SMA_L)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
        self.data.dropna(inplace = True)
    
        order_info = {}
        target_profit = 1.1
        stop_loss = 0.85
        # sma crossover strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            date, price = self.get_values(bar)
            
            if len(order_info) > 0:
                target_profit_price = order_info["target_profit"]
                stop_loss_price = order_info["stop_loss"]
                buy_price = order_info["buy_price"]
            
                if price > target_profit_price:
                    print(f"SELL: Target Profit hit || bought_price: {buy_price}, target profit: {target_profit_price}")
                    self.go_short(bar) # go short with full amount
                    self.position = 0 # short position
                    order_info = {}
                    continue
                elif price < stop_loss_price:
                    self.go_short(bar) # go short with full amount
                    self.position = 0 # short position
                    print(f"SELL: Stop Loss hit || bought_price: {buy_price}, stop loss: {stop_loss_price}")
                    order_info = {}
                    continue
                    
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                if self.position == 0:
                    self.go_long(bar) # go long with full amount
                    self.position = 1  # long position
                    
                    stop_loss_price = 0
                    target_profit_profit = 0
                    
                    stop_loss_price = price * stop_loss
                    target_profit_price = price * target_profit
                    
                    order_info["buy_price"] = price
                    order_info["target_profit"] = target_profit_price
                    order_info["stop_loss"] = stop_loss_price
                    print(f"Stop loss: {stop_loss_price}")
                    print(f"Target: {target_profit_price}")
            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                if self.position == 1:
                    self.go_short(bar) # go short with full amount
                    self.position = 0 # short position
                    order_info = {}
        self.close_pos(bar+1) # close position at the last bar
        
    def test_sma_strategy(self, SMA_S, SMA_L):
        
        # nice printout
        stm = "Testing SMA strategy | {} | SMA_S = {} & SMA_L = {}".format(self.symbol, SMA_S, SMA_L)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
        self.data.dropna(inplace = True)
    

        # sma crossover strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            date, price = self.get_values(bar)
                    
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                if self.position == 0:
                    self.go_long(bar) # go long with full amount
                    self.position = 1  # long position

            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                if self.position == 1:
                    self.go_short(bar) # go short with full amount
                    self.position = 0 # short position
                    
        self.close_pos(bar+1) # close position at the last bar        
        
import os 
parent_dir = "hist_data"     
symbol = "ETH"    
path = os.path.join(parent_dir,  str(symbol + "/5min/"+symbol+"_5min.csv"))

bc = IterativeBacktest(symbol, "2021-08-12 19:05:00", "2021-10-08 21:55:00", 3300, path)
#bc.test_sma_strategy_bracket(65,158)
bc.test_macd_strategy_bracket(19, 23, 10)

