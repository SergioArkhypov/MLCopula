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
        """
        Initializes the Market data acquisition object with portfolio parameters and retrieval settings.

        Args:
            name (str): A string identifier for the market or portfolio dataset (e.g., "SP500_Long").
            cob_date_str (str): The close-of-business (COB) end date for the data retrieval, formatted as 'YYYYMMDD'.
            ticker_list (list): A list of stock ticker symbols (strings) to fetch historical data for.
            period_years (int or float): The total number of years of historical data to retrieve, looking backward from the COB date.
            interval (str): The data frequency or interval expected by the yfinance API (e.g., '1d' for daily data).
            cache_path (str): The local directory path where the retrieved market data will be saved and cached as a CSV file.
        """
        self.ticker_list = ticker_list
        self.period_years = period_years
        self.interval = interval
        self.name = name
        self.cob_date_str = cob_date_str
        self.cache_path = cache_path


    def load_history(self):
        """
        Loads historical market data from a local cache or fetches it via the Yahoo Finance API.

        This method first checks if a cached CSV file exists for the specified portfolio and 
        timeframe. If found, it loads the data directly into a pandas DataFrame. If not, it 
        iterates through the ticker list, retrieves the 'Close' prices for the specified 
        period using yfinance, forward-fills any missing values, and saves the consolidated 
        DataFrame to the local cache. Finally, it extracts the most recent prices and stores 
        them as the current spot prices.

        Attributes Set:
            self.market (pd.DataFrame): The consolidated historical price data for all tickers.
            self.spot (pd.Series): The most recent observed prices (last row) from the dataset.
        """

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
        """
        Extracts a specific historical sub-period and calculates the log-returns for the portfolio.

        This method filters the previously loaded market data to isolate a timeframe ending on 
        the specified close-of-business (COB) date and extending backward by a given number of years. 
        It then computes the log-returns for all tickers within this window. Data integrity is 
        maintained by dropping rows where all data points are missing and filling any remaining 
        null values with zero.

        Args:
            cob_date_str (str): The end date for the return calculation window, formatted as 'YYYYMMDD'.
            sub_period_years (int or float): The number of years of data to include in the calculation, 
                                            looking backward from the cob_date_str.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated log-returns for the specified sub-period, 
                        indexed by date.
        """
        end_date = datetime.strptime(cob_date_str, '%Y%m%d')
        start_date = end_date - timedelta(days=sub_period_years*365)
        start_date_str = start_date.strftime('%Y%m%d')
        market = self.market[(self.market.index > start_date_str) & (self.market.index <= cob_date_str)]

        returns = np.log(1.0 + market.pct_change(fill_method = None)) 
        returns = returns.dropna(how='all')
        returns = returns.fillna(0)
        print(f'Market stats {cob_date_str}-{sub_period_years}Y, non nones count: {sum(list(returns.count()))}, needs {returns.size}')
        return returns