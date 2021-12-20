'''
Created on 14.12.2021

@author: GP4EYN2
'''


from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from IterativeBase2 import IterativeBase


class IterativeBacktest(IterativeBase):

    # helper method
    def go_long(self, bar): 
        self.buy_instrument(bar) # go long

    # helper method
    def go_short(self, bar, ): 

        self.sell_instrument(bar) # go short
        
    
    def store_position_data(self, price):
        
        stop_loss = round(price - (price * self.stop_loss_pct), 2)
        take_profit = round(price + (price * self.take_profit_pct), 2)     
        self.position_info['profit'] = take_profit
        self.position_info['loss'] = stop_loss
        #print(f"New position data || {price}, stop_loss: {stop_loss}, take_profit: {take_profit}")
        
        
    def compute_sma_parameters(self):
        
        self.cur_df["EMA_S"] = self.cur_df["Close"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean()
        self.cur_df["EMA_L"] = self.cur_df["Close"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
        self.cur_df["MACD"] = self.cur_df.EMA_S - self.cur_df.EMA_L    
        self.cur_df["MACD_Signal"] = self.cur_df.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
        self.cur_df.dropna(inplace = True)
        
        
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
        #self.get_data() # reset dataset
        
        # prepare data
        
        if EMA_S is not None:
            self.EMA_S = EMA_S
                        
        if EMA_L is not None:
            self.EMA_L = EMA_L
        
        if signal_mw is not None:
            self.signal_mw = signal_mw

        # sma crossover strategy
        test_data_len = len(self.df) - self.end_idx - 1
        if test_data_len > 0:
            for bar in range(test_data_len): # all bars (except the last bar)
                self.update_current_df()
                self.compute_sma_parameters()
                #_, price = self.get_values(bar)  
                
                price = self.cur_df["Close"][-1]    
                if self.position: 
                    # Try to neutral
                    if price <= self.position_info["loss"]:
                        print(f"Stop loss triggered.")
                        #print(f"Going neutral from long for {price} $")
                        self.go_short(price)
                        self.position = 0  # neutral position
                        self.trades += 1   
                        
                    elif price >= self.position_info["profit"]:
                        print(f"Take profit triggered.")
                        #print(f"Going neutral from long for {price} $")
                        self.go_short(price)
                        self.position = 0  # neutral position
                        self.trades += 1
                        
                    elif self.cur_df["MACD"].iloc[-1] < self.cur_df["MACD_Signal"].iloc[-1]: # signal to go short
                        #print(f"Going neutral from long for: {price} $")
                        self.go_short(price)
                        self.position = 0  # neutral position
                        self.trades += 1
                else:
                    if self.cur_df["MACD"].iloc[-1] > self.cur_df["MACD_Signal"].iloc[-1]: # signal to go long
                        #print(f"Going long from neutral for: {price} $")
                        self.position_info = {}
                        self.store_position_data(price)
                        self.go_long(price) # go long with full amount
                        self.position = 1  # long position
                        self.trades += 1
            self.close_pos(self.df["Close"].iloc[-1]) # close position at the last bar   
        else: 
            print("Not enough data for backtest..")
        
        
        
        
if __name__ == "__main__":        
    import os 
    parent_dir = "hist_data"     
    symbol = "ETH"    
    path = os.path.join(parent_dir,  str(symbol + "/5min/"+symbol+"_5min.csv"))
    
    bc = IterativeBacktest(symbol, "2021-08-12 19:05:00", "2021-10-08 21:55:00", 3300, path)
    #bc.test_sma_strategy_bracket(65,158)
    bc.test_macd_strategy_bracket(19, 23, 10)
    
        
        
        
        
        
        
        
        
        
        