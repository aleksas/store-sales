import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import download_csv, makedirs, join_path, ZipFile, exists

# Example usage
base_url = 'https://github.com/aleksas/store-sales/releases/download/data/'
files = [
    'train.csv.zip',
    'transactions.csv.zip',
    'test.csv',
    'stores.csv',
    'oil.csv',
    'holidays_events.csv',
]

dir = './data/'
makedirs(dir, exist_ok=True)

for file in files:
    path = join_path(dir, file)
    if not exists(path):
        download_csv(join_path(base_url, file), path)
        if path.endswith('.zip'):
            with ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(dir)

# Load data
train = pd.read_csv(join_path(dir, 'train.csv'), parse_dates=['date'])
test = pd.read_csv(join_path(dir, 'test.csv'), parse_dates=['date'])
stores = pd.read_csv(join_path(dir, 'stores.csv'))
oil = pd.read_csv(join_path(dir, 'oil.csv'), parse_dates=['date'])
holidays = pd.read_csv(join_path(dir, 'holidays_events.csv'), parse_dates=['date'])

# Merge additional data
train = pd.merge(train, stores, on='store_nbr')
test = pd.merge(test, stores, on='store_nbr')

train.rename(columns={'type': 'store_type'}, inplace=True)
test.rename(columns={'type': 'store_type'}, inplace=True)

train = pd.merge(train, oil, on='date', how='left')
test = pd.merge(test, oil, on='date', how='left')

holidays = holidays[holidays['transferred'] == False]

train = pd.merge(train, holidays[['date', 'type']], on='date', how='left')
test = pd.merge(test, holidays[['date', 'type']], on='date', how='left')

train.rename(columns={'type': 'holiday_type'}, inplace=True)
test.rename(columns={'type': 'holiday_type'}, inplace=True)

# Fill missing oil price values
train['dcoilwtico'].fillna(method='ffill', inplace=True)
test['dcoilwtico'].fillna(method='ffill', inplace=True)

# Feature engineering
train['is_holiday'] = (train['holiday_type'].notna()).astype(int)
test['is_holiday'] = (test['holiday_type'].notna()).astype(int)

train['is_payday'] = ((train['date'].dt.day == 15) | (train['date'].dt.day == train['date'].dt.days_in_month)).astype(int)
test['is_payday'] = ((test['date'].dt.day == 15) | (test['date'].dt.day == test['date'].dt.days_in_month)).astype(int)

earthquake_date = pd.to_datetime('2016-04-16')
train['post_earthquake'] = ((train['date'] > earthquake_date) & (train['date'] <= earthquake_date + pd.DateOffset(weeks=4))).astype(int)
test['post_earthquake'] = ((test['date'] > earthquake_date) & (test['date'] <= earthquake_date + pd.DateOffset(weeks=4))).astype(int)

# One-hot encoding
train = pd.get_dummies(train, columns=['holiday_type', 'city', 'state', 'store_type', 'cluster'])
test = pd.get_dummies(test, columns=['holiday_type', 'city', 'state', 'store_type', 'cluster'])

# Align train and test data
train, test = train.align(test, join='outer', axis=1, fill_value=0)

# Normalize data
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train.drop(['date', 'store_nbr', 'family', 'sales'], axis=1))
scaled_test = scaler.transform(test.drop(['date', 'store_nbr', 'family', 'sales'], axis=1))

# Create target variable
y = train['sales'].values
y_test = test['sales'].values

# Split the data
X_train, X_val, y_train, y_val = train_test_split(scaled_train, y, test_size=0.2, shuffle=False)

