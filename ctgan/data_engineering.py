import pandas as pd
import numpy as np


def data_engineering(df):
    date_columns = ['creation_date', 'view_date', 'action_date']
    
    for c in date_columns:
        df[c] = df[c].apply(lambda x: x[:16])
    
    df[date_columns] = df[date_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')