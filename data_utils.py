import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(file_path='Steel_industry_data.csv'):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear

    df = pd.get_dummies(df, columns=['WeekStatus', 'Day_of_week', 'Load_Type'], drop_first=True)
    return df


def create_supervised_dataset(data, target_cols, feature_cols, time_steps=24):

    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    X_scaled = scaler_features.fit_transform(data[feature_cols])
    y_scaled = scaler_targets.fit_transform(data[target_cols])

    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(X_scaled[i:(i + time_steps)])
        y.append(y_scaled[i + time_steps])

    return np.array(X), np.array(y), scaler_features, scaler_targets


def create_multiscale_multitask_dataset(data, targets, features, short_steps, long_steps):

    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    X_scaled = scaler_features.fit_transform(data[features])
    y_scaled = scaler_targets.fit_transform(data[targets])

    X_short, X_long, Y = [], [], []

    for i in range(len(data) - long_steps):
        X_short.append(X_scaled[i + long_steps - short_steps: i + long_steps])
        X_long.append(X_scaled[i: i + long_steps])
        Y.append(y_scaled[i + long_steps])

    return [np.array(X_short), np.array(X_long)], np.array(Y), scaler_features, scaler_targets