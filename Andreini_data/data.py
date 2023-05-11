import requests
import json
import pandas as pd
import datetime as dt
from tqdm import tqdm
import pickle
import warnings
import numpy as np
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
KEY = "51692ad112e439586cfe21f4fb436f50"

#Define transform as specified in Andreini
def apply_lagpoly(x):
    """
    compute (1-L)(1-L^12)X_t
    """
    res = x
    res = res[12:]-res[:-12]
    res = res[1:] - res[:-1]
    res = np.append([np.nan]*13, res)
    return res

TRANSFORMS = {
    0: lambda x: np.append([np.nan],x[1:] - x[:-1]), #there is one TCode 0 in the appendix which is not defined
    1: lambda x: x,
    2: lambda x: np.append([np.nan],x[1:] - x[:-1]),
    3: apply_lagpoly,
    4: lambda x: np.log(x),
    5: lambda x: np.append([np.nan], np.log(x)[1:] - np.log(x)[:-1]),
    6: lambda x: apply_lagpoly(np.log(x)),
}

def apply_transform(dfs: dict, codes: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transforms as specified in appendix of Andreini
    """
    for i, row in codes.iterrows():
        id = row['Code']
        if id in dfs.keys():
            transform  = int(row['TCode'])
            print(dfs[id].columns)
            dfs[id][id] = TRANSFORMS[transform](dfs[id][id].values)
        else:
            print(f"Could not find {id} in columns when applying transforms")
    return dfs

def fetch_series(id: str):
    response = requests.get(f"https://api.stlouisfed.org/fred/series/observations",
                    headers = {'Accept': 'application/json'},
                    params = {"series_id": id,
                            "api_key": KEY,
                            "file_type": "json"})
                            #,
                            #"observation_start": START_STRING,
                            #"observation_end": END_STRING})
    if response.status_code == 200:
        data = json.loads(response.content)['observations']
    else:
        raise ValueError(f"{id} could not be fetched: {response.content}")
    
    df = pd.DataFrame(data)[['date', 'value']]
    df.columns = ['date', id]
    df[id] = df[id].astype('float64')
    df = df.set_index('date', drop=True)
    return df

def fetch_all_series(codes:list[str]) -> dict:
    res = {}
    err = ""
    for code in tqdm(codes):
        try:
            df = fetch_series(code)
            res[code] = df
        except ValueError as e:
            err += str(e)+ '\n'
            continue
        time.sleep(0.01)#avoid request limit
    print(err)
    return res

def merge_dfs(dfs: dict):
    df_conc = pd.concat(dfs, axis=1).sort_values(by='date')
    return df_conc

def load_y(daterange = ['1973-01-01', '2023-01-01']):
    df = pd.read_csv('../Andreini_data/data_transformed.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    #interpolate gpdc column
    df['GDPC1'] = df['GDPC1'].interpolate(method='spline', order=1)
    df = df.loc[daterange[0]:daterange[1]]
    #df = df.dropna(axis=1, how='any')
    y = df.values
    mask = (~np.isnan(y)).astype('float')
    #y= (y-y.min(axis=0))/(y.max(axis=0)-y.min(axis=0))
    y = y - y.mean(axis=0)
    y = y / np.std(y,axis=0)
    return y, mask, df.index

if __name__ == "__main__":
    codes = pd.read_csv('codes.txt', sep=' ')
    codes_fred = codes[codes['Source'] == 'FRED']
    print(len(codes_fred))
    dfs = fetch_all_series(codes_fred['Code'])
    dfs = apply_transform(dfs, codes_fred)
    print(len(dfs))
    print(f"Fetched data for {len(dfs.keys())} codes: {list(dfs.keys())} ")
    df_conc = merge_dfs(dfs)
    df_conc.to_csv('data_transformed.csv')

