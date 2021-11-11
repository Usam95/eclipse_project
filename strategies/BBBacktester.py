
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")


class BBBacktester():
    ''' Class for the vectorized backtesting of Mean Reversion-based trading strategies (Bollinger Bands).

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    SMA: int
        time window for SMA
    dev: int
        distance for Lower/Upper Bands in Standard Deviation units
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
        sets one or two new parameters for SMA and dev
        
    test_strategy:
        runs the backtest for the Mean Reversion-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the two parameters
    '''
    
    def __init__(self, symbol, df, SMA, dev, tc, duration, bar_size):
        self.symbol = symbol
        self.SMA = SMA
        self.dev = dev
        self.tc = tc
        self.df = df
        self.results = None
        self.get_data()
        self.name = "BB"
        
        self.perf = None
        self.outperf = None
        
        self.bar_size =  bar_size        
        self.duration = duration
        
    def __repr__(self):
        rep = "BBBacktester(symbol = {}, SMA = {}, dev = {}, start = {}, end = {})"
        return rep.format(self.symbol, self.SMA, self.dev)
    
    def get_hist_data_params(self):
        return (self.name, self.symbol, self.duration, self.bar_size)
    
    def get_parameters(self): 
        
        return (self.SMA, self.dev, 0)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        tmp_df = pd.DataFrame()
        tmp_df['Close'] = self.df['Close']
        tmp_df["returns"] = np.log(tmp_df['Close'] / tmp_df['Close'].shift(1))
        tmp_df["SMA"] = tmp_df["Close"].rolling(self.SMA).mean()
        tmp_df["Lower"] = tmp_df["SMA"] - tmp_df["Close"].rolling(self.SMA).std() * self.dev
        tmp_df["Upper"] = tmp_df["SMA"] + tmp_df["Close"].rolling(self.SMA).std() * self.dev
        self.data = tmp_df
        
    def set_parameters(self, SMA = None, dev = None):
        ''' Updates parameters and resp. time series.
        '''
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["Close"].rolling(self.SMA).mean()
            self.data["Lower"] = self.data["SMA"] - self.data["Close"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["Close"].rolling(self.SMA).std() * self.dev
            
        if dev is not None:
            self.dev = dev
            self.data["Lower"] = self.data["SMA"] - self.data["Close"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["Close"].rolling(self.SMA).std() * self.dev
            
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["distance"] = data.Close - data.SMA
        data["position"] = np.where(data.Close < data.Lower, 1, np.nan)
        data["position"] = np.where(data.Close > data.Upper, 0, data["position"])
        data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])
        data["position"] = data.position.ffill().fillna(0)
        data["strategy"] = data.position.shift(1) * data["returns"]
        data.dropna(inplace = True)
        
        # determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
       
        # absolute performance of the strategy
        perf = data["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - data["creturns"].iloc[-1]
        
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
            title = "{} | BB: SMA = {} | dev = {} | TC = {}".format(self.symbol, self.SMA, self.dev, self.tc)
            return self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
      
    def getPerfData(self): 
        if self.perf !=None and self.outperf!=None: 
            return self.perf, self.outperf
        else:
            return 0, 0
        
    def update_and_run(self, boll):
        ''' Updates parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        Params: tuple
            parameter tuple with SMA and dist
        '''
        self.set_parameters(int(boll[0]), int(boll[1]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, SMA_range, dev_range):
        ''' Finds global maximum given the parameter ranges.

        Parameters
        ==========
        SMA_range, dist_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (SMA_range, dev_range), finish=None)
        self.test_strategy()
        return opt, -self.update_and_run(opt)
