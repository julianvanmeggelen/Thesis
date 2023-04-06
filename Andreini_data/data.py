import requests
import json
import pandas as pd
import datetime as dt
from tqdm import tqdm
import pickle
import warnings
import numpy as np
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

def apply_transform(df_conc: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transforms as specified in appendix of Andreini
    """
    for i, row in codes.iterrows():
        id = row['Code']
        if id in df_conc.columns:
            transform  = int(row['TCode'])
            df_conc[id] = TRANSFORMS[transform](df_conc[id].values)
        else:
            print(f"Could not find {id} in columns when applying transforms")
    return df_conc

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
    print(err)
    return res

def merge_dfs(dfs: dict):
    df_conc = pd.concat(dfs, axis=1).sort_values(by='date')
    return df_conc

if __name__ == "__main__":
    codes = pd.read_csv('codes.txt', sep=' ')
    codes_fred = codes[codes['Source'] == 'FRED']
    print(len(codes_fred))
    dfs = fetch_all_series(codes_fred['Code'])
    print(len(dfs))
    print(f"Fetched data for {len(dfs.keys())} codes: {list(dfs.keys())} ")
    df_conc = merge_dfs(dfs)
    df_conc.to_csv('data_raw.csv')
    df_conc = apply_transform(df_conc, codes_fred)
    df_conc.to_csv('data_transformed.csv')

