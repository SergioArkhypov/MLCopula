import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class Market:
    """
    To execute the complete workflow, we start with the portfolio information provided as an input.
    Subsequently, we access historical stock price data from Yahoo Finance API, according to the 
    specified timeframe duration. The implementation performs storing of all retrieved data in a
    locally saved CSV file for the reference in case multiple runs be necessary. After the 
    gathering stage of all required timeseries, they undergo pre-processing to calculate
    log-returns, segmented into the chunks of dedicated length.
    """

    def __init__(self, name, cob_date_str, ticker_list, period_years, interval, cache_path):
        self.ticker_list = ticker_list
        self.period_years = period_years
        self.interval = interval
        self.name = name
        self.cob_date_str = cob_date_str
        self.cache_path = cache_path


    def load_history(self):
        market_path = f'{self.cache_path}\\market-{self.name}-{self.cob_date_str}-{self.period_years}Y-{self.interval}.csv'

        if os.path.exists(market_path):
            self.market = pd.read_csv(market_path, index_col=0)
            self.market.index = self.market.index.map(str)

        else:    
            market = []
            for ticker_symbol in self.ticker_list:
                ticker = yf.Ticker(ticker_symbol)
                end_date = datetime.strptime(self.cob_date_str, '%Y%m%d').date()
                start_date = end_date - timedelta(days=self.period_years*365)
                historical_data = ticker.history(start=start_date, end=end_date, interval=self.interval)  # data for the last year
                historical_data = historical_data[['Close']]
                new_index = pd.Index(list(map(lambda o: o.strftime('%Y%m%d'), historical_data.index))) #list(map(lambda k: ,df.index))
                historical_data = historical_data.set_index(new_index)
                historical_data = historical_data.rename(columns={'Close': ticker_symbol}, errors="raise")
                market.append(historical_data)

            market = pd.concat(market, axis=1)
            market = market.sort_index(ascending=True)
            market = market.ffill()
            market.to_csv(market_path)
            self.market = market
        
        self.spot = self.market.iloc[-1]


    def get_logreturns(self, cob_date_str, sub_period_years):
        end_date = datetime.strptime(cob_date_str, '%Y%m%d')
        start_date = end_date - timedelta(days=sub_period_years*365)
        start_date_str = start_date.strftime('%Y%m%d')
        market = self.market[(self.market.index > start_date_str) & (self.market.index <= cob_date_str)]

        returns = np.log(1.0 + market.pct_change(fill_method = None)) 
        returns = returns.dropna(how='all')
        returns = returns.fillna(0)
        print(f'Market stats {cob_date_str}-{sub_period_years}Y, non nones count: {sum(list(returns.count()))}, needs {returns.size}')
        return returns