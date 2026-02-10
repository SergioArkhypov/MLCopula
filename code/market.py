import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def get_market(ticker_list, period, interval):

    market = []
    for ticker_symbol in ticker_list:
        ticker = yf.Ticker(ticker_symbol)

        historical_data = ticker.history(period=period, interval=interval)  # data for the last year
        historical_data = historical_data[['Close']]
        new_index = pd.Index(list(map(lambda o: o.strftime("%Y%m%d"), historical_data.index))) #list(map(lambda k: ,df.index))
        historical_data = historical_data.set_index(new_index)
        historical_data = historical_data.rename(columns={'Close': ticker_symbol}, errors="raise")
        market.append(historical_data)

    result = pd.concat(market, axis=1)
    result = result.sort_index(ascending=True)
    #result = np.log(result).shift()
    returns = np.log(1.0 + result.pct_change(fill_method = None)) #TODO: implement other 
    return (returns.dropna(), result.iloc[-1])