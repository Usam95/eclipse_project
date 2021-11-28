'''
Created on 14.11.2021

@author: usam
'''


from datetime import date 
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime 
from datetime import time
import calendar

import pandas as pd
import os
from pandas._libs.tslibs import period

import fxcmpy

from time import sleep

class HistData: 
    
    def __init__(self, crypto, symbol, period):
        
        if symbol is None or period is None: 
            print("Symbol and Period must be defined!")
        self.symbol = symbol
        self.period = period
        self.crypto = crypto
        
        self.create_path()
    def create_path(self):
        folder = str("./hist_data/"+self.crypto+"/"+self.period)
        if os.path.exists(folder): 
            self.path = str(folder+"/"+self.crypto+"_"+self.period+".csv")
            print(self.path)
        else:
            print(f"ERROR: The directory {folder} does not exist.")
            exit(0)   
                                        
        
    def getHistData(self, api, months=5):
        print("Called getHistData.")

        period_str = self.period[0]
        amount = self.period[1:]
        amount = int(amount)
        hour = 0
        minutes = 0
        if period_str == "H": 
            
            hour = 24 - (amount)
            minutes = 0
        elif period_str == "m":
            minutes = 60 - (amount)
            hour = 23
            
        df_final = pd.DataFrame()
        
        start = date.today() + relativedelta(months=-months)
        month_first_start = start.replace(day=1)
        
        month_first_start = datetime.combine(month_first_start, time(0,0,0))
        
        month = month_first_start.month
        for i in range(months):
            #month = 5
            #print(f"month: {month}")
            month_range = calendar.monthrange(2021, month)
            #print(f"month_range: {month_range}")
            month_end = month_range[1]
            #print(f"month_end: {month_end}")
            #month_first_start = datetime(2021, month, 1, 0, 0, 0)
            #print(f"month_first_start: {month_first_start}")
            month_first_end = month_first_start + timedelta(days=14, hours=hour, minutes=minutes)
            
            df_first = api.get_candles(self.symbol, start = month_first_start, end = month_first_end, 
                        period = self.period, columns = ["asks"])
            
            print("Called get_candles for df_first.")
            print(f"Lenght of first df: {len(df_first)}")

            #sleep(5)
            #print(f"month_firs_end: {month_first_end}")
            month_second_start = month_first_start + timedelta(days=15)
            #print(f"month_second_start: {month_second_start}")
        
            month_second_end = datetime(2021, month, month_end, hour, minutes, 0)
            
            df_second = api.get_candles(self.symbol, start = month_second_start, end = month_second_end, 
                        period = self.period, columns = ["asks"])
            print("Called get_candles for df_second.")
            print(f"Lenght of second df: {len(df_second)}")

            #sleep(5)
            #time.sleep(5)

            #print(f"month_second_end: {month_second_end}")
            month += 1
            month_first_start = month_first_start + timedelta(days=month_end)
            df_final.append(df_first)
            df_final.append(df_second)
        
        print(f"Storing data with length: {len(df_final)} to {self.path}")
        if len(df_final) > 0: 
            df_final.to_scv(self.path)        
            print(f"Stored data with length: {len(df_final)} to {self.path}")
        else: 
            print(f"No data stored for {self.path}.")
            
            
if __name__ == "__main__":
     
    crypto_dic = {"LTC":"LTC/USD", "EOS":"EOS/USD", "BCH":"BCH/USD"}
    periods = ["m1", "m5", "m15", "m30", "H1", "H2", "H3", "H4"]
    months = 1
    api = fxcmpy.fxcmpy(config_file= "fxcm.cfg")
    if api.is_connected():
        try:
            print("api is connected.")
            for crypto, symbol in crypto_dic.items(): 
                for period in periods: 
                    hist_data_obj = HistData(crypto, symbol, period)
                    print("hist_data_obj is created.")
                    hist_data_obj.getHistData(api, months)
        except: 
            print("Error during processing hist data. ")
            api.close()
    else: 
        print("ERROR: Couldn't connect to fxcm server.")
    api.close()
    exit(0)