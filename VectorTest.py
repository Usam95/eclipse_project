#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 13:44:25 2021

@author: usam
"""

# Import  IB libraries

#import other libraries
import pandas as pd
import numpy as np 
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os 

from strategies.RSIBacktester import RSIBacktester 
from strategies.EMABacktester import EMABacktester 
from strategies.BBBacktester import BBBacktester
from strategies.SMABacktester import SMABacktester
from strategies.MACDBacktester import MACDBacktester
from strategies.SOBacktester import SOBacktester

backtest_data_length = 0.8
forwardtest_data_length = 0.2
ptc = 0.00007
pp = PdfPages('FR_Plot.pdf')

parent_dir = "./hist_data/5M/"
   

def get_backtest_data(df): 
    return df.iloc[:len(df)*backtest_data_length, :]

def get_forwardtest_data(df):
    return df.iloc[:len(df)*forwardtest_data_length, :]


#file_names = os.listdir(parent_dir)

def save_figure(strategy):
    plot = strategy.plot_results()
    fig = plot.get_figure()
    pp.savefig(fig)
    plt.close(fig)
    

duration = 5
bar_size = 2 

def comb_strategy_2(symbol, strategy_1, strategy_2): 
    df = pd.DataFrame()
    df['returns'] = strategy_1.data['returns']
    
    pos_strg_1 = "position" + strategy_1.name
    pos_strg_2 = "position" + strategy_2.name
    
    df[pos_strg_1] = strategy_1.results['position'].astype("bool")
    df[pos_strg_2] = strategy_2.results['position'].astype("bool")

    df['returns'] = strategy_1.data['returns']
    df.dropna(inplace=True)
    df['comb_position'] = (df[pos_strg_1] & df[pos_strg_2]).astype("int")
    df['comb_position'] = df['comb_position'].shift(1)
    df = df.fillna(0)
    df.dropna(inplace=True)

    df['strategy'] = (df['comb_position'] * df['returns'])
    df.dropna(inplace=True)

    df['trades'] = df.comb_position.diff().fillna(0).abs()
    df.strategy = df.strategy - df.trades * ptc
    
    df['creturns'] = df['returns'].cumsum().apply(np.exp)
    df['cstrategy'] = df['strategy'].cumsum().apply(np.exp)    
    value_counts = df["comb_position"].value_counts()
    print(f"Value Counts: {value_counts}")
    print(f"Return buy hold: {df['creturns'][-1]}")
    print(f"Return strategy: {df['cstrategy'][-1]}")
    #fig = plt.figure()
    title = "Combination of strategies: {} || {} || for {}".format(strategy_1.name.upper(), strategy_2.name.upper(), symbol)
    plot = df[['creturns', 'cstrategy']].plot(title = title, figsize=(12,8), fontsize=12)
    fig = plot.get_figure()
    pp.savefig(fig)
    plt.close(fig)
    print("========================================================")
    print("========================================================")
    
def comb_strategy_3(symbol, strategy_1, strategy_2, strategy_3): 
    df = pd.DataFrame()
    df['returns'] = strategy_1.data['returns']
    
    pos_strg_1 = "position" + strategy_1.name
    pos_strg_2 = "position" + strategy_2.name
    pos_strg_3 = "position" + strategy_3.name
    
    df[pos_strg_1] = strategy_1.results['position'].astype("bool")
    df[pos_strg_2] = strategy_2.results['position'].astype("bool")
    df[pos_strg_3] = strategy_3.results['position'].astype("bool")

    df['returns'] = strategy_1.data['returns']
    df.dropna(inplace=True)
    df['comb_position'] = (df[pos_strg_1] & df[pos_strg_2] & df[pos_strg_3]).astype("int")
    df['comb_position'] = df['comb_position'].shift(1)
    df = df.fillna(0)
    df.dropna(inplace=True)

    df['strategy'] = (df['comb_position'] * df['returns'])
    df.dropna(inplace=True)

    df['trades'] = df.comb_position.diff().fillna(0).abs()
    df.strategy = df.strategy - df.trades * ptc
    
    df['creturns'] = df['returns'].cumsum().apply(np.exp)
    df['cstrategy'] = df['strategy'].cumsum().apply(np.exp)    
    value_counts = df["comb_position"].value_counts()
    print(f"Value Counts: {value_counts}")
    print(f"Return buy hold: {df['creturns'][-1]}")
    print(f"Return strategy: {df['cstrategy'][-1]}")
    #fig = plt.figure()
    title = "Combination of strategies: {} || {} || {} for {}".format(strategy_1.name.upper(), strategy_2.name.upper(), strategy_3.name.upper(), symbol)
    plot = df[['creturns', 'cstrategy']].plot(title = title, figsize=(12,8), fontsize=12)
    fig = plot.get_figure()
    pp.savefig(fig)
    plt.close(fig)
    print("========================================================")
    print("========================================================")


def test_strategies(symbol_list):

    parent_dir = "./hist_data/"
    # + symbol + "/5min/"+symbol+"_5min.csv"
    
    files = os.listdir(parent_dir)
    
    for file_name in files: 
        symbol = file_name
        if symbol == "ETH":
            path = os.path.join(parent_dir,  str(symbol + "/5min/"+symbol+"_5min.csv"))
            df = pd.read_csv(path)
            df.rename(columns={"date": "Date", "askopen": "Open", "askclose": "Close", "askhigh": "High", "asklow": "Low"}, errors="raise", inplace=True)
            df.set_index("Date",inplace=True)
        
            print("Processing the symbol: ", symbol)
            ema = EMABacktester(symbol, df, 50, 200, ptc, duration, bar_size)
            #ema.optimize_parameters((25,75,1), (100,200,1))
            ema.set_parameters(58, 146)
            print("EMA test results: ", ema.test_strategy())
            #print("EMA parameters: ", ema.get_parameters())
            save_figure(ema) 
    
    # df['Close'].plot(figsize=(12,8))
    # rsi_tester = RSIBacktester(symbol, df = df, periods = 20, rsi_upper = 70, rsi_lower = 30, tc = ptc)
    # rsi_tester.optimize_parameters((5, 20, 1), (65, 80, 1), (20, 35, 1))
    # print("RSI: ", rsi_tester.test_strategy())
    # save_figure(rsi_tester)

            rsi = RSIBacktester(symbol, df, 20, 70, 30, ptc, duration, bar_size)
            #rsi.optimize_parameters((5, 20, 1), (65, 80, 1), (20, 35, 1))
            rsi.set_parameters(15, 65, 28)
            print("RSI test results: ", rsi.test_strategy())
            #print("RSI parameters: ", rsi.get_parameters())

            save_figure(rsi)
            
            
            bb = BBBacktester(symbol, df, 30, 2, ptc, duration, bar_size)
            bb.set_parameters(30, 3)
            #bb.optimize_parameters((25, 100, 1), (1,5,1))
            print("BB test results: ", bb.test_strategy())
            #print("BB parameters: ", bb.get_parameters())
            save_figure(bb)
            
            
            sma = SMABacktester(symbol, df, 50, 200, ptc)
            sma.set_parameters(65, 158)
            #sma.optimize_parameters((25,75,1), (100,200,1))
            print("SMA test results: ", sma.test_strategy())
            #print("SMA parameters: ", sma.get_parameters())
            save_figure(sma)
    
            macd = MACDBacktester(symbol, df, 12, 26, 9, ptc)
            #macd.set_parameters(65, 158,1)
            #(19, 23, 10)
            macd.optimize_parameters((5,20,1), (21,50,1), (5,20,1))
            print("MACD test results: ", macd.test_strategy())
            print("MACD parameters: ", macd.get_parameters())
            save_figure(macd)
            
            so = SOBacktester(symbol, df, 14, 3, ptc)
            #so.set_parameters(14,19)
            so.optimize_parameters((10,100,1), (3,50,1))
            print("SO test results: ", so.test_strategy())
            print("SO parameters: ", so.get_parameters())
            save_figure(so)
    # ema_tester = EMABacktester(symbol, df, 50, 200, ptc)
    # ema_tester.optimize_parameters((25, 75, 1), (100, 200, 1))
    # print("EMA: ", ema_tester.test_strategy())
    # save_figure(ema_tester)
    
    # so_tester = SOBacktester(symbol, df, 14, 3, ptc)
    # so_tester.optimize_parameters((10, 100, 1), (3, 50, 1))
    # print("SO: ", so_tester.test_strategy())
    # save_figure(so_tester))
    
    # bb_teter = BBBacktester(symbol, df, 30, 2, ptc)
    # bb_teter.optimize_parameters((25, 100, 1), (1, 5, 1))
    # print("BB: ", bb_teter.test_strategy())
    # save_figure(bb_teter)    
        # fr_teter = FRBacktester(symbol, df, 70, ptc)
        # print("BB: ", fr_teter.test_strategy())
        # save_figure(fr_teter)       
            
        
            comb_strategy_3(symbol,  ema, rsi, bb)
            comb_strategy_2(symbol,  ema, rsi)
            comb_strategy_2(symbol,  ema, bb)
            comb_strategy_2(symbol,  rsi, bb)
            
            comb_strategy_2(symbol,  sma, rsi)
            comb_strategy_2(symbol,  sma, bb)
            comb_strategy_2(symbol,  macd, rsi)
            comb_strategy_2(symbol,  macd, bb)
            comb_strategy_2(symbol,  macd, so)

    pp.close()

symbol_lst = ["ETH"]
test_strategies(symbol_lst)

