import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CovidDataset
from sklearn.preprocessing import MinMaxScaler

def get_features_df(lat_eps:float, long_eps:float, country: str = 'Poland', filename: str = 'time_series_covid19_confirmed_global.csv', ):
    df = pd.read_csv(filename, index_col=False)

    lat = df[df['Country/Region'] == country][['Lat', 'Long']].values[0][0]
    long = df[df['Country/Region'] == country][['Lat', 'Long']].values[0][1]

    df = df[((df['Lat'] <= lat + lat_eps) & (df['Lat'] >= lat - lat_eps)) & (
                (df['Long'] <= long + long_eps) & (df['Long'] >= long - long_eps))]

    df['Province/State'] = df['Province/State'].fillna(df['Country/Region'])

    df.index = df['Province/State']
    df.drop(['Province/State', 'Country/Region'], inplace=True, axis=1)

    df = df.T
    df = df.drop(['Lat', 'Long'])
    df = df.reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week
    df["quarter"] = df["date"].dt.quarter

    cols = ["date", "year", "month", "day", "dayofweek", "week", "quarter"] + [col for col in df.columns if
                                                                               col not in ["date", "year", "month",
                                                                                           "day", "dayofweek", "week",
                                                                                           "quarter"]]
    df = df[cols]

    features_df = df.drop(columns=["date"])


    return features_df, df['date'].tolist()


def train_test_split(features_df:pd.DataFrame, train_size: float = 0.9):
    train_size = int(len(features_df) * train_size)
    train_df, test_df = features_df[:train_size], features_df[train_size:]


    return train_df, test_df

def get_scaled_dfs(scaler, train_df, test_df):
    train_df = pd.DataFrame(scaler.transform(train_df), index=train_df.index, columns=train_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), index=test_df.index, columns=test_df.columns)
    return train_df, test_df
def create_sequences(input_data, target_column, sequence_length):

    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - sequence_length)):
        sequence = input_data[i:i+sequence_length]
        label = input_data.iloc[i+sequence_length][target_column]

        sequences.append((sequence, label))

    return sequences

def create_datasets(train_sequence, test_sequence):
    train_dataset = CovidDataset(train_sequence)
    test_dataset = CovidDataset(test_sequence)

    return train_dataset, test_dataset

def create_loaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

