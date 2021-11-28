# =============================================================================
# Automated trading script I - MACD
# Author : Mayank Rasu

# Please report bug/issues in the Q&A section
# =============================================================================

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


#===============================================================================
# import smtplib
# from email.MIMEMultipart import MIMEMultipart
# from email.MIMEBase import MIMEBase
# from email import Encoders
#===============================================================================

#defining strategy parameters

# crypto currency to be traded in the strategy
#pairs = ['LTC/USD','ETH/USD']

pairs = ['ETH/USD']

#max capital allocated/position size for each cryptocurrency
pos_size = {'LTC/USD':5, 'ETH/USD':1}


class RenkoMacd:
    
    def __init__(self, cryptos, pos_size):
        self.cryptos = cryptos 
        
        self.con = self.connect()
        self.cryptos = cryptos
        self.pos_size = pos_size
        
        self.macd_str = MACDTester(EMA_S=12, EMA_L=26, signal_mw=9)
        
        self.macd_params = {}
        
        self.logger = None
        
        #initialiue log file
        self.logSetup(create_file=True)
        self.log_file_name = None
        
        self.macd_update_time = 12
        
#===============================================================================
#     def logSetup(self, create_file = True):
#         #self.log_file = open(self.path, "w")
#         #TODO: check if the file was open
#         #TODO: use log file system
#         #TODO: 1. send current file with gmail, 
#         #      2. arhives the file, 
#         #      3. create new log file
#         
# 
#         if self.logger: 
#             handlers = self.log.handlers[:]
#             for handler in handlers:
#                 handler.close()
#                 self.log.removeHandler(handler)
#                 
#         self.log_file_name = datetime.now().strftime('logs/logfile_%H_%M_%S_%d_%m_%Y.log')
#         
#         self.logger = logging.getLogger(self.log_file_name)
#         self.logger.setLevel(logging.INFO)
#===============================================================================

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
        con = fxcmpy.fxcmpy(config_file= "fxcm.cfg", log_level = 'info', server='demo')
        print("Connected to fxcm server...")
        #self.logger.info("Connected to fxcm server...")

        #TODO: check if connection was established
        return con
    

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
    
    def trade_signal(self, MERGED_DF,l_s):
        "function to generate signal"
        signal = ""
        df = copy.deepcopy(MERGED_DF)
        if l_s == "":
            if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
                signal = "Buy"
            elif df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
                signal = "Sell"
                
        elif l_s == "long":
            if df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
                signal = "Close_Sell"
            elif df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
                signal = "Close"
                
        elif l_s == "short":
            if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
                signal = "Close_Buy"
            elif df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
                signal = "Close"
        return signal
      
    def optimize_macd_params(self):
        
        if self.macd_update_time >=12: 
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
   
                 
                else: 
                    self.logger.info(f"Not candles data was received for crypto: {crypto}")
            self.macd_update_time = 0  
        else:  
            self.macd_update_time +=1
        #=======================================================================
        # self.macd_params['LTC/USD'] = (12,26,9)
        # self.macd_params['ETH/USD'] = (12,26,9)
        #=======================================================================

    def postprocess(self):
        
        now = dt.now()
        if now.hour == 0 and now.minute == 0 and now.seconds == 0: 
            self.send_email()
            
            self.logSetup(create_file=True)
            
            
    def execute_strategy(self):
        
        open_pos = self.con.get_open_positions()
        for crypto in self.cryptos:
            long_short = ""
            if len(open_pos)>0:
                open_pos_cur = open_pos[open_pos["currency"]==crypto]
                if len(open_pos_cur)>0:
                    if open_pos_cur["isBuy"].tolist()[0]==True:
                        long_short = "long"
                    elif open_pos_cur["isBuy"].tolist()[0]==False:
                        long_short = "short"   
            data = self.con.get_candles(crypto, period='m5', number=250)
            ohlc = data.iloc[:,[0,1,2,3,8]]
            ohlc.columns = ["Open","Close","High","Low","Volume"]
            
            close_price = ohlc["Close"][-1]
            signal = self.trade_signal(self.renko_merge(ohlc, crypto),long_short)
    
            if signal == "Buy":
                trade_obj = self.con.open_trade(symbol=crypto, is_buy=True, amount=pos_size[crypto], 
                               time_in_force='GTC', order_type='AtMarket')
                #self.logger.info("New long position initiated for ", crypto)
                self.report_trade("Buy", crypto, close_price)
            elif signal == "Sell":
                trade_obj = self.con.open_trade(symbol=crypto, is_buy=False, amount=pos_size[crypto], 
                               time_in_force='GTC', order_type='AtMarket')
                self.report_trade("Sell", crypto, close_price)
                #self.logger.info("New short position initiated for ", crypto)
                
            elif signal == "Close":
                self.con.close_all_for_symbol(crypto)
                self.logger.info("\n" + 100* "-")
                self.logger.info("All positions closed for ", crypto)
                self.logger.info("\n" + 100* "-")
            elif signal == "Close_Buy":
                self.con.close_all_for_symbol(crypto)
                self.logger.info("\n" + 100* "-")
                self.logger.info("Existing Short position closed for ", crypto)
                trade_obj = self.con.open_trade(symbol=crypto, is_buy=True, amount=pos_size[crypto], 
                               time_in_force='GTC', order_type='AtMarket')
                #self.logger.info("New long position initiated for ", crypto)
                self.report_trade("Buy", crypto, close_price)
                
            elif signal == "Close_Sell":
                self.con.close_all_for_symbol(crypto)
                self.logger.info("\n" + 100* "-")
                self.logger.info("Existing long position closed for ", crypto)
                trade_obj = self.con.open_trade(symbol=crypto, is_buy=False, amount=pos_size[crypto], 
                               time_in_force='GTC', order_type='AtMarket')
                
                
                self.logger.info("New short position initiated for ", crypto)
                #self.report_trade("Going Short", crypto, close_price)
                self.report_trade("Sell", crypto, close_price)
                
    def report_trade(self, going, crypto, price):
        open_pos_df = self.con.get_open_positions()
        if (len(open_pos_df) > 0):
            price = open_pos_df.open.iloc[-1]
            unreal_pl = open_pos_df.grossPL.sum()
        
            self.logger.info("{} for crypto {}".format(going, crypto))
            self.logger.info("price = {} | Unreal. P&L = {}".format(price, unreal_pl))
            self.logger.info(100 * "-" + "\n")   
        #optimize parameters for next run
        


    def send_email(self):
        msg = MIMEMultipart('alternative')
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:           
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    
            toEmail, fromEmail = EMAIL_ADDRESS, EMAIL_ADDRESS
            msg['Subject'] = 'FXCM Report'
            msg['From'] = fromEmail
            body = "The report is attached to this email as file."
        
            # add log file 
            
        
            f = open(self.logger.name, 'r')
            attachment = MIMEText(f.read())
            attachment.add_header('Content-Disposition', 'attachment', filename=self.logger.name)           
            msg.attach(attachment)
            f.close()
            
            content = MIMEText(body, 'plain')
            msg.attach(content)
            
            smtp.sendmail(fromEmail, toEmail, msg.as_string())
        
def main(strategy):
    try:
        strategy.execute_strategy()
        strategy.optimize_macd_params()
    except:
        print("error encountered....skipping this iteration")
# Continuous execution  
      
starttime=time.time()

strategy = RenkoMacd(pairs, pos_size)
strategy.logger.info("Starting execute strategy.")

#strategy.report_trade("SELL", "ETH", "0.187")
#strategy.send_email()
 
if strategy.con.is_connected():
    strategy.optimize_macd_params()
   
   
    # TODO: optimize macd params for each currency -> put in a function                   
    timeout = time.time() + 60*60*24*7  # 60 seconds times 60 meaning the script will run for 1 hr
    while time.time() <= timeout:
        try:
            print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            main(strategy)
            time.sleep(300 - ((time.time() - starttime) % 300.0)) # 5 minute interval between each new execution
        except KeyboardInterrupt:
            strategy.logger.error('\n\nKeyboard exception received. Exiting.')
            exit()
       
    # Close all positions and exit
    for currency in pairs:
        strategy.logger.info(f"closing all positions for {currency}.")
        strategy.con.close_all_for_symbol(currency)
    strategy.con.close()
else: 
    strategy.logger.info("Could not connected to fxcm server...")

