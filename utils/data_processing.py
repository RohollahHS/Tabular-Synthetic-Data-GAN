import pandas as pd
import numpy as np
# import seaborn as sns
# sns.set_style("darkgrid")

def preprocessing(file_name, data_path):
    df = pd.read_csv(f'{data_path}/{file_name}.csv')

    date_columns = ['creation_date', 'view_date', 'action_date']

    for c in date_columns:
        df[c] = df[c].apply(lambda x: x[:16])

    df[date_columns] = df[date_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')

    df['view_creation_distance'] = df['view_date'] - df['creation_date']
    df['action_view_distance'] = df['action_date'] - df['view_date']

    df['view_creation_distance_minute'] = df['view_creation_distance'].dt.total_seconds().div(60).astype(int)

    df['action_view_distance_minute'] = df['action_view_distance'].dt.total_seconds().div(60).astype(int)

    df['creation_date_year'] = df['creation_date'].dt.year
    df['creation_date_month'] = df['creation_date'].dt.month
    df['creation_date_day'] = df['creation_date'].dt.day
    df['creation_date_hour'] = df['creation_date'].dt.hour
    df['creation_date_minute'] = df['creation_date'].dt.minute

    df['delta_creation_date'] = df['creation_date'].sort_values().diff().dt.total_seconds().div(60).fillna(0).astype(int)

    drop_columns = ['creation_date', 'view_date', 'action_date', 'view_creation_distance', 'action_view_distance']
    df_dropped = df.drop(columns=drop_columns)

    df_dropped.to_csv(f'{data_path}/preprocessed_{file_name}.csv', index=False)
    
    return df_dropped


def post_processing(synthetic_data, file_name, data_path):
    indexs = synthetic_data[synthetic_data['creation_date_month'] <= 0]['creation_date_month'].index
    synthetic_data.loc[indexs, 'creation_date_month'] = 1
    indexs = synthetic_data[synthetic_data['creation_date_month'] >= 13]['creation_date_month'].index
    synthetic_data.loc[indexs, 'creation_date_month'] = 12

    indexs = synthetic_data[synthetic_data['creation_date_day'] <= 0]['creation_date_day'].index
    synthetic_data.loc[indexs, 'creation_date_day'] = 1
    indexs = synthetic_data[synthetic_data['creation_date_day'] >= 32]['creation_date_day'].index
    synthetic_data.loc[indexs, 'creation_date_day'] = 31

    indexs = synthetic_data[synthetic_data['creation_date_hour'] < 0]['creation_date_hour'].index
    synthetic_data.loc[indexs, 'creation_date_hour'] = 0
    indexs = synthetic_data[synthetic_data['creation_date_hour'] >= 24]['creation_date_hour'].index
    synthetic_data.loc[indexs, 'creation_date_hour'] = 0

    indexs = synthetic_data[synthetic_data['creation_date_minute'] < 0]['creation_date_minute'].index
    synthetic_data.loc[indexs, 'creation_date_minute'] = 0
    indexs = synthetic_data[synthetic_data['creation_date_minute'] >= 60]['creation_date_minute'].index
    synthetic_data.loc[indexs, 'creation_date_minute'] = 59


    indexes = synthetic_data[synthetic_data['view_creation_distance_minute'] < 0]['view_creation_distance_minute'].index
    synthetic_data.loc[indexes, 'view_creation_distance_minute'] = 0

    indexes = synthetic_data[synthetic_data['action_view_distance_minute'] < 0]['action_view_distance_minute'].index
    synthetic_data.loc[indexes, 'action_view_distance_minute'] = 0


    def create_creation_date(year, month, day, hour, minute, second=0):
        return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

    synthetic_data['creation_date_str'] = synthetic_data.apply(lambda row: create_creation_date(
                                                            row['creation_date_year'], 
                                                            row['creation_date_month'], 
                                                            row['creation_date_day'], 
                                                            row['creation_date_hour'], 
                                                            row['creation_date_minute']
                                                            ),axis=1)

    for i in range(synthetic_data.shape[0]):
        try:
            synthetic_data.loc[i, 'creation_date'] = pd.to_datetime(synthetic_data['creation_date_str'].iloc[i])
        except:
            rand = np.random.randint(1, 25)
            date = list(synthetic_data.iloc[i]['creation_date_str'])
            date[8:11] = list(f'{rand:02d} ')
            synthetic_data.loc[i, 'creation_date'] = pd.to_datetime(''.join(date))
            

    synthetic_data['view_creation_distance_minute_delta'] = pd.to_timedelta(synthetic_data['view_creation_distance_minute'], unit='m')
    synthetic_data['action_view_distance_minute_delta'] = pd.to_timedelta(synthetic_data['action_view_distance_minute'], unit='m')

    synthetic_data['view_date'] = synthetic_data['creation_date'] + synthetic_data['view_creation_distance_minute_delta']
    synthetic_data['action_date'] = synthetic_data['view_date'] + synthetic_data['action_view_distance_minute_delta']


    drop_columns = [
        'view_creation_distance_minute',
        'action_view_distance_minute',
        'creation_date_year',
        'creation_date_month',
        'creation_date_day',
        'creation_date_hour',
        'creation_date_minute',
        'delta_creation_date',
        'creation_date_str',
        'view_creation_distance_minute_delta',
        'action_view_distance_minute_delta'
    ]

    final_data = synthetic_data.drop(columns=drop_columns)

    synthetic_data.to_csv(f'{data_path}/raw_synthetic_{file_name}.csv', index=False)
    final_data.to_csv(f'{data_path}/fianl_synthetic_{file_name}.csv', index=False)

    return final_data, synthetic_data
