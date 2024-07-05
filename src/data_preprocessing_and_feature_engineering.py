import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_data(train_path, test_path, store_path):
    dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')
    types = {'StateHoliday': str, 'SchoolHoliday': str}
    
    df_train = pd.read_csv(train_path, parse_dates=['Date'], date_parser=dateparse, dtype=types)
    df_test = pd.read_csv(test_path, parse_dates=['Date'], date_parser=dateparse, dtype=types)
    df_store = pd.read_csv(store_path)
    
    return df_train, df_test, df_store

def clean_data(df_train, df_test, df_store):
    # Handle missing values in store data
    df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace=True)
    df_store['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df_store['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df_store['Promo2SinceWeek'].fillna(0, inplace=True)
    df_store['Promo2SinceYear'].fillna(0, inplace=True)
    df_store['PromoInterval'].fillna('0', inplace=True)
    
    # Remove outliers and invalid data points from training data
    df_train = df_train[(df_train['Open'] != 0) & (df_train['Sales'] > 0)].copy()
    df_train = df_train.reset_index(drop=True)
    
    # Merge datasets
    train = pd.merge(df_train, df_store, how='left', on='Store')
    test = pd.merge(df_test, df_store, how='left', on='Store')
    
    return train, test

def extract_features(data):
    features = ['Store', 'CompetitionDistance']
    
    # Extract date features
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek + 1
    data['WeekOfYear'] = data.Date.dt.isocalendar().week
    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    
    # Competition and Promo Open features
    data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['Month'] - data['CompetitionOpenSinceMonth'])
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    features.extend(['CompetitionOpen', 'PromoOpen'])
    
    # Label encode categorical features
    categorical_features = ['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 'Promo', 'Promo2']
    features.extend(categorical_features)
    for column in categorical_features:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
    
    # Encode PromoInterval
    PromoInterval_maps = {'0': 1, 'Jan,Apr,Jul,Oct': 2, 'Feb,May,Aug,Nov': 3, 'Mar,Jun,Sept,Dec': 4}
    data.PromoInterval = data.PromoInterval.map(PromoInterval_maps)
    features.append('PromoInterval')
    
    return data, features

if __name__ == "__main__":
    train, test, store = load_data("../data/train.csv", "../data/test.csv", "../data/store.csv")
    train_cleaned, test_cleaned = clean_data(train, test, store)
    train_processed, features = extract_features(train_cleaned)
    test_processed, _ = extract_features(test_cleaned)
    
    train_processed.to_csv("../data/train_processed.csv", index=False)
    test_processed.to_csv("../data/test_processed.csv", index=False)
    print("Data preprocessing complete. Processed files saved.")
