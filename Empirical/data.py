import yfinance as yf
import numpy as np
import pandas as pd
URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tickers = pd.read_html(URL)[0]['Symbol'].tolist()



def getSp500Data(start= '1920-01-01', end= '2023-01-01', f=None):
    f = f"./dsp500{start}_{end}.csv"
    data = yf.download(tickers, start=start, end=end)
    data.to_csv(f)

def getLogRet():
    f = 'dsp5001990-01-01_2023-01-01.csv'
    data = pd.read_csv(f, index_col=0)
    cols = [col for col in data.columns if 'Adj Close' in col]
    data = data[cols]
    vals = data.iloc[2:].values.astype('float64')
    logprice = np.log(vals, where=~np.isnan(vals))
    logret = logprice[1:] - logprice[:-1]
    mask = np.isnan(vals)[1:]
    return logret, mask

if __name__ == "__main__":
    getSp500Data()