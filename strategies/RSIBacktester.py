
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")

class RSIBacktester(): 
    ''' Class for the vectorized backtesting of RSI-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    periods: int
        time window in days to calculate moving average UP & DOWN 
    rsi_upper: int
        upper rsi band indicating overbought instrument
    rsi_lower: int
        lower rsi band indicating oversold instrument
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade
        
        
    Methods
    =======
    get_data:
        retrieves and prepares the data
        
    set_parameters:
        sets new RSI parameter(s)
        
    test_strategy:
        runs the backtest for the RSI-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates RSI parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the three RSI parameters
    '''
    
    def __init__(self, symbol, df, periods, rsi_upper, rsi_lower, tc, duration, bar_size):
        self.symbol = symbol
        self.periods = periods
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.tc = tc
        self.results = None 
        self.df = df
        self.get_data()
        self.name = "RSI"
        self.perf = None
        self.outperf = None
        
        self.bar_size = bar_size        
        self.duration = duration
    def __repr__(self):
        return "RSIBacktester(symbol = {}, RSI({}, {}, {}), start = {}, end = {})".format(self.symbol, self.periods, self.rsi_upper, self.rsi_lower)
        
    
    def get_hist_data_params(self):
        return (self.name, self.symbol, self.duration, self.bar_size)
    
    def get_parameters(self): 
        
        return (self.periods, self.rsi_upper, self.rsi_lower)
    
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        tmp_df = pd.DataFrame()
        tmp_df['Close'] = self.df['Close']
        tmp_df["returns"] = np.log(tmp_df['Close'] / tmp_df['Close'].shift(1))
        tmp_df["U"] = np.where(tmp_df['Close'].diff() > 0, tmp_df['Close'].diff(), 0) 
        tmp_df["D"] = np.where(tmp_df['Close'].diff() < 0, -tmp_df['Close'].diff(), 0)
        tmp_df["MA_U"] = tmp_df.U.rolling(self.periods).mean()
        tmp_df["MA_D"] = tmp_df.D.rolling(self.periods).mean()
        tmp_df["RSI"] = tmp_df.MA_U / (tmp_df.MA_U + tmp_df.MA_D) * 100
        self.data = tmp_df
        
    def set_parameters(self, periods = None, rsi_upper = None, rsi_lower = None):
        ''' Updates RSI parameters and resp. time series.
        '''
        if periods is not None:
            self.periods = periods     
            self.data["MA_U"] = self.data.U.rolling(self.periods).mean()
            self.data["MA_D"] = self.data.D.rolling(self.periods).mean()
            self.data["RSI"] = self.data.MA_U / (self.data.MA_U + self.data.MA_D) * 100
            
        if rsi_upper is not None:
            self.rsi_upper = rsi_upper
            
        if rsi_lower is not None:
            self.rsi_lower = rsi_lower
            
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data.RSI > self.rsi_upper, 0, np.nan)
        data["position"] = np.where(data.RSI < self.rsi_lower, 1, data.position)
        data.position = data.position.fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc

        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
        self.perf = round(perf, 6)
        self.outperf = round(outperf, 6)
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | RSI ({}, {}, {}) | TC = {}".format(self.symbol, self.periods, self.rsi_upper, self.rsi_lower, self.tc)
            return self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
      
    def getPerfData(self): 
        if self.perf !=None and self.outperf!=None: 
            return self.perf, self.outperf
        else:
            return 0, 0
        
    def update_and_run(self, RSI):
        ''' Updates RSI parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        RSI: tuple
            RSI parameter tuple
        '''
        self.set_parameters(int(RSI[0]), int(RSI[1]), int(RSI[2]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, periods_range, rsi_upper_range, rsi_lower_range):
        ''' Finds global maximum given the RSI parameter ranges.

        Parameters
        ==========
        periods_range, rsi_upper_range, rsi_lower_range : tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (periods_range, rsi_upper_range, rsi_lower_range), finish=None)
        self.test_strategy()
        return opt, -self.update_and_run(opt)
    
    