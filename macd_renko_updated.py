'''
Created on 05.12.2021

@author: usam
'''
import fxcmpy
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import time
import copy

import logging

from datetime import datetime
from strategies.MACDBacktester import MACDBacktester as MACDTester
import strategies

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from locale import currency

EMAIL_ADDRESS = "usam.sersultanov@gmail.com"
EMAIL_PASSWORD = "gqpzqjfmkhgjmpjn"


from datetime import datetime as dt
from pytz import timezone

#defining strategy parameters

# crypto currency to be traded in the strategy
pairs = ['LTC/USD','ETH/USD', 'BCH/USD']

#pairs = ['ETH/USD']

#max capital allocated/position size for each cryptocurrency
pos_size = {'LTC/USD':5, 'ETH/USD':2, 'BCH/USD':2}

class RenkoMacd:
    
    def __init__(self, cryptos, pos_size):
        self.cryptos = cryptos 
        
        self.connect()
        self.cryptos = cryptos
        self.pos_size = pos_size
        
        self.macd_str = MACDTester(EMA_S=12, EMA_L=26, signal_mw=9)
        
        self.macd_params = {
            'LTC/USD': (12,26,9), 
            'ETH/USD': (12,26,9), 
            'BCH/USD': (12,26,9)
            }
        
        self.logger = None
        
        #initialiue log file
        self.logSetup(create_file=True)
        self.log_file_name = None
        
        self.macd_update_time = 36
        
        self.create_file = True
        self.tk_profit = 0.05

    def logSetup(self, create_file=False):
    
            # create logger for prd_ci
            self.log_file_name = datetime.now().strftime('logs/logfile_%d_%m_%Y.log')
            
            log = logging.getLogger(self.log_file_name)
            log.setLevel(level=logging.INFO)
    
            # create formatter and add it to the handlers
            formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(funcName)-16s %(message)s',
                                          datefmt='%m-%d-%y %H:%M:%S')
            if create_file:
                # create file handler for logger.
                fh = logging.FileHandler(self.log_file_name)
                fh.setLevel(level=logging.INFO)
                fh.setFormatter(formatter)
            # reate console handler for logger.
            ch = logging.StreamHandler()
            ch.setLevel(level=logging.INFO)
            ch.setFormatter(formatter)
    
            # add handlers to logger.
            if create_file:
                log.addHandler(fh)
    
            log.addHandler(ch)
            self.logger = log 
        
    def connect(self):
        self.con = fxcmpy.fxcmpy(config_file= "fxcm.cfg", log_level = 'info', server='demo')
        

    def MACD(self, DF, crypto):
        """function to calculate MACD
           typical values a = 12; b =26, c =9"""
        df = DF.copy()
        
        a,b,c = self.macd_params[crypto]
        
        df["MA_Fast"]=df["Close"].ewm(span=a,min_periods=a).mean()
        df["MA_Slow"]=df["Close"].ewm(span=b,min_periods=b).mean()
        df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
        df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
        df.dropna(inplace=True)
        return (df["MACD"],df["Signal"])
    
    def ATR(self, DF,n):
        "function to calculate True Range and Average True Range"
        df = DF.copy()
        df['H-L']=abs(df['High']-df['Low'])
        df['H-PC']=abs(df['High']-df['Close'].shift(1))
        df['L-PC']=abs(df['Low']-df['Close'].shift(1))
        df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
        df['ATR'] = df['TR'].rolling(n).mean()
        #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
        df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
        return df2
    
    def slope(self, ser,n):
        "function to calculate the slope of n consecutive points on a plot"
        slopes = [i*0 for i in range(n-1)]
        for i in range(n,len(ser)+1):
            y = ser[i-n:i]
            x = np.array(range(n))
            y_scaled = (y - y.min())/(y.max() - y.min())
            x_scaled = (x - x.min())/(x.max() - x.min())
            x_scaled = sm.add_constant(x_scaled)
            model = sm.OLS(y_scaled,x_scaled)
            results = model.fit()
            slopes.append(results.params[-1])
        slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
        return np.array(slope_angle)
    
    def renko_DF(self, DF):
        "function to convert ohlc data into renko bricks"
        df = DF.copy()
        df.reset_index(inplace=True)
        df = df.iloc[:,[0,1,2,3,4,5]]
        df.columns = ["date","open","close","high","low","volume"]
        df2 = Renko(df)
        df2.brick_size = round(self.ATR(DF,120)["ATR"][-1],4)
        renko_df = df2.get_ohlc_data()
        renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
        for i in range(1,len(renko_df["bar_num"])):
            if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
                renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
            elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
                renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
        return renko_df
    
    def renko_merge(self, DF, crypto):
        "function to merging renko df with original ohlc df"
        df = copy.deepcopy(DF)
        df["Date"] = df.index
        renko = self.renko_DF(df)
        renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
        merged_df = df.merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
        merged_df["bar_num"].fillna(method='ffill',inplace=True)
        merged_df["macd"]= self.MACD(merged_df, crypto)[0]
        merged_df["macd_sig"]= self.MACD(merged_df, crypto)[1]
        merged_df["macd_slope"] = self.slope(merged_df["macd"],5)
        merged_df["macd_sig_slope"] = self.slope(merged_df["macd_sig"],5)
        return merged_df
    
    def execute_strategy(self):
        
        for crypto in self.cryptos:
                print("starting passthrough for.....", crypto)
                
                data = self.con.get_candles(crypto, period='m5', number=150)
                ohlc = data.iloc[:,[0,1,2,3,8]]
                ohlc.columns = ["Open","Close","High","Low","Volume"]
            
                price = ohlc["Close"][-1]            
                
                df = self.renko_merge(ohlc, crypto)
                
                open_pos_df = self.con.get_open_positions()
                order_df = self.con.get_orders()
                atr = self.ATR(ohlc, 60)["ATR"][-1]
                
                st_loss = round(atr,1)
                tk_profit = round(price * self.tk_profit,1)
                
                if len(order_df) > 0: 
                    self.logger.info("In 1: len(order_df) > 0: ")
                    self.logger.info(f"Number of orders: {len(order_df)}")
                    if order_df[order_df['currency']==crypto]:
                        
                        order_id = order_df[order_df['currency']==crypto]['orderId']
                        order = self.con.get_order(order_id)
                        if order.get_status() == 'waiting': 
                            self.con.change_order_stop_limit(order_id, stop=st_loss, limit=tk_profit)
                        self.logger.info(f"Updated Stop Loss for available order for {crypto}..")
                
                if len(open_pos_df)==0:
                    self.logger.info("In 2: len(open_pos_df)==0:")
                    #self.logger.info(f"Number of positions: {len(open_pos_df)}")

                    if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:

                        self.con.open_trade(symbol = crypto, is_buy = True,
                                           amount = pos_size[crypto],
                                           time_in_force = 'GTC', stop = st_loss, limit = tk_profit,
                                           trailing_step = False, order_type = 'AtMarket')
                        
                        self.report_trade("Going long from neutral", crypto, price)
                                    
                elif len(open_pos_df)!=0 and len(open_pos_df[open_pos_df['currency'] == crypto]) == 0:
                    self.logger.info("In 3: len(open_pos_df)!=0 and len(open_pos_df[open_pos_df['currency'] == crypto]) == 0:")
                    self.logger.info(f"Number of positions: {len(open_pos_df)}")

                    if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
                        
                        self.con.open_trade(symbol = crypto, is_buy = True,
                                           amount = pos_size[crypto],
                                           time_in_force = 'GTC', stop = st_loss, limit = tk_profit,
                                           trailing_step = False, order_type = 'AtMarket')
                         
                        self.report_trade("Going long from neutral", crypto, price)
                       
                elif len(open_pos_df)!=0 and len(open_pos_df[open_pos_df['currency'] == crypto]) > 0:

                    self.logger.info("In 4: len(open_pos_df)!=0 and len(open_pos_df[open_pos_df['currency'] == crypto]) > 0:")
                    self.logger.info(f"Number of positions: {len(open_pos_df)}")

                    if df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
                        self.con.open_trade(symbol=crypto, is_buy=False, amount=pos_size[crypto], time_in_force='GTC', order_type='AtMarket')
                        self.report_trade("Going neutral from long", crypto, price)
                    else: 
                        self.con.change_trade_stop_limit(trade_id = open_pos_df[open_pos_df['currency'] == crypto]['tradeId'], is_in_pips = False, is_stop = True, rate = st_loss)
                        self.logger.info(f"Updated Stop Loss for open position for:{crypto}..")
                        


                        
    def report_trade(self, msg, crypto, price):
        time.sleep(2)

        day_pl = self.con.get_accounts_summary()['dayPL'][0]
        self.logger.info("{} for crypto {}".format(msg, crypto))
        self.logger.info("Price = {} | Dayly. P&L = {}".format(price, day_pl))
        self.logger.info(100 * "-" + "\n")   
        
    def optimize_macd_params(self):
        
        start_t = time.time()
        if self.macd_update_time >= 36: 
            for crypto in self.cryptos: 
                self.macd_str.set_symbol(crypto)  
                df = self.con.get_candles(crypto, period='m5', number=9000)
                if len(df) > 0:
                    df = df.iloc[:,[0,1,2,3,8]]
                    df.columns = ["Open","Close","High","Low","Volume"]
                     
                    #print(f"Lenght of df: {len(df)}")
                    self.macd_str.set_df(df)
                    self.macd_str.optimize_parameters((5,20,1), (21,50,1), (5,20,1))
                    params = self.macd_str.get_parameters()
                    self.macd_params[crypto] = params
                    self.logger.info(f"Updated parameters for {crypto}|| a: {params[0]}, b: {params[1]}, c: {params[2]}")
                    perf, outperf = self.macd_str.test_strategy()
                    self.logger.info(f"Perf: {perf}, Outperf: {outperf}.")
                 
                else: 
                    self.logger.info(f"Not candles data was received for crypto: {crypto}")
            self.macd_update_time = 0 
            end_t = time.time()
            dur_t = end_t - start_t
            self.logger.info(f"Total time took to update parameters: {dur_t/60} mins.")
        else:  
            self.macd_update_time += 1
            
            
strategy = RenkoMacd(pairs, pos_size)
strategy.logger.info("Strategy initialized. Trying to start program execution.")

#strategy.report_trade("SELL", "ETH", "0.187")
#strategy.send_email()

def main(strategy):

    strategy.execute_strategy()
    strategy.optimize_macd_params()



starttime=time.time()        
timeout = time.time() + 60*60*24*7  # 60 seconds times 60 meaning the script will run for 1 hr
first_pass = True
while time.time() <= timeout:
    if strategy.con.is_connected():
        
        if first_pass: 
            
            open_pos_df = strategy.con.get_open_positions()
            if len(open_pos_df) > 0: 
                strategy.con.close_all()
                strategy.logger.info(f"Closed {len(open_pos_df)} open positions.")
            
            order_ids_df = strategy.con.get_order_ids()
            if len(order_ids_df) > 0: 
                for order_id in order_ids_df: 
                    strategy.con.delete_order(order_id)
                strategy.logger.info(f"Closed {len(order_id)} open orders.")
            
            strategy.logger.info("First pass: optimizing parameters..")   
            strategy.optimize_macd_params()
            first_pass = False
        # TODO: optimize macd params for each currency -> put in a function           

        try:
            print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            main(strategy)
            time.sleep(300 - ((time.time() - starttime) % 300.0)) # 5 minute interval between each new execution
        except KeyboardInterrupt:
            strategy.logger.error('\n\nKeyboard exception received. Exiting.')
            strategy.logger.error("Error occured during strategy execution. Wait 1 minute and try again.")
            strategy.con.close()
            exit()
        except: 
            strategy.logger.error("Error occured during strategy execution. Wait 1 minute and try again.")
            strategy.con.close()
            time.sleep(30)
            strategy.connect()
            time.sleep(30)
        # Close all positions and exit
        #===========================================================================
        # for currency in pairs:
        #     strategy.logger.info(f"closing all positions for {currency}.")
        #     strategy.con.close_all_for_symbol(currency)
        # strategy.con.close()
        #===========================================================================
    else: 
        strategy.logger.error('Connection issue, reseting connection.')
        strategy.con.close()
        time.sleep(30)
        strategy.connect()
                   