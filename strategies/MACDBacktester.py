
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")

class MACDBacktester(): 
    ''' Class for the vectorized backtesting of MACD-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    EMA_S: int
        time window in days for shorter EMA
    EMA_L: int
        time window in days for longer EMA
    signal_mw: int
        time window is days for MACD Signal 
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
        sets new MACD parameter(s)
        
    test_strategy:
        runs the backtest for the MACD-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates MACD parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the three MACD parameters
    '''
    
    def __init__(self, symbol, df, EMA_S, EMA_L, signal_mw, tc, **kwargs):
        self.symbol = symbol
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.signal_mw = signal_mw
        self.tc = tc
        self.results = None 
        self.df = df
        self.get_data()
        self.name = "MACD"
    def __repr__(self):
        return "MACDBacktester(symbol = {}, MACD({}, {}, {}), start = {}, end = {})".format(self.symbol, self.EMA_S, self.EMA_L, self.signal_mw)
     
    def get_parameters(self): 
        
        return (self.EMA_S, self.EMA_L, self.signal_mw)

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        #raw = pd.read_csv("forex_pairs.csv", parse_dates = ["Date"], index_col = "Date")
        #raw = raw[self.symbol].to_frame().dropna()
        #raw = raw.loc[self.start:self.end]
        tmp_df = pd.DataFrame()
        tmp_df['Close'] = self.df['Close']
        tmp_df["returns"] = np.log(tmp_df['Close'] / tmp_df['Close'].shift(1))
        tmp_df["EMA_S"] = tmp_df["Close"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean() 
        tmp_df["EMA_L"] = tmp_df["Close"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
        tmp_df["MACD"] = tmp_df.EMA_S - tmp_df.EMA_L
        tmp_df["MACD_Signal"] = tmp_df.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()   
        
        self.data = tmp_df
        
    def set_parameters(self, EMA_S = None, EMA_L = None, signal_mw = None):
        ''' Updates MACD parameters and resp. time series.
        '''
        if EMA_S is not None:
            self.EMA_S = EMA_S
            self.data["EMA_S"] = self.data["Close"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean()
            self.data["MACD"] = self.data.EMA_S - self.data.EMA_L
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
            
        if EMA_L is not None:
            self.EMA_L = EMA_L
            self.data["EMA_L"] = self.data["Close"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()
            self.data["MACD"] = self.data.EMA_S - self.data.EMA_L
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
            
        if signal_mw is not None:
            self.signal_mw = signal_mw
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean()
            
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        
        data = self.data.copy().dropna()
        data["position"] = np.where(data["MACD"] > data["MACD_Signal"], 1, 0)
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
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | MACD ({}, {}, {}) | TC = {}".format(self.symbol, self.EMA_S, self.EMA_L, self.signal_mw, self.tc)
            return self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
        
    def update_and_run(self, MACD):
        ''' Updates MACD parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        MACD: tuple
            MACD parameter tuple
        '''
        self.set_parameters(int(MACD[0]), int(MACD[1]), int(MACD[2]))
        return -self.test_strategy()[0]
    
    def optimize_parameters(self, EMA_S_range, EMA_L_range, signal_mw_range):
        ''' Finds global maximum given the MACD parameter ranges.

        Parameters
        ==========
        EMA_S_range, EMA_L_range, signal_mw_range : tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (EMA_S_range, EMA_L_range, signal_mw_range), finish=None)
        return opt, -self.update_and_run(opt)
    
    