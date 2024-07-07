import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(train_path, test_path, store_path):
    # Load data
    dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')
    df_train = pd.read_csv(train_path, parse_dates=['Date'], date_parser=dateparse)
    df_test = pd.read_csv(test_path, parse_dates=['Date'], date_parser=dateparse)
    df_store = pd.read_csv(store_path)
    
    # Merge data
    train = pd.merge(df_train, df_store, how='left', on='Store')
    test = pd.merge(df_test, df_store, how='left', on='Store')
    
    # Clean data
    for df in [train, test]:
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
        df['Promo2SinceWeek'].fillna(0, inplace=True)
        df['Promo2SinceYear'].fillna(0, inplace=True)
        df['PromoInterval'].fillna('0', inplace=True)
    
    # Remove invalid data points from train
    train = train[(train['Open'] != 0) & (train['Sales'] > 0)].copy()
    
    return train, test

def engineer_features(df):
    # Time-based features
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['DayOfWeek'] = df.Date.dt.dayofweek
    df['WeekOfYear'] = df.Date.dt.isocalendar().week
    
    # Cyclical encoding of time features
    df['MonthSin'] = np.sin(2 * np.pi * df.Month/12)
    df['MonthCos'] = np.cos(2 * np.pi * df.Month/12)
    df['DayOfWeekSin'] = np.sin(2 * np.pi * df.DayOfWeek/7)
    df['DayOfWeekCos'] = np.cos(2 * np.pi * df.DayOfWeek/7)
    
    # Competition features
    df['CompetitionOpen'] = 12 * (df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth)
    df['CompetitionOpen'] = df.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    
    # Promo features
    df['PromoOpen'] = 12 * (df.Year - df.Promo2SinceYear) + (df.WeekOfYear - df.Promo2SinceWeek) / 4.0
    df['PromoOpen'] = df.PromoOpen.apply(lambda x: x if x > 0 else 0)
    df.loc[df.Promo2SinceYear == 0, 'PromoOpen'] = 0
    
    # Interaction features
    df['Promo_StoreType'] = df['Promo'].astype(str) + '_' + df['StoreType']
    
    # Label encoding
    cat_features = ['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 'PromoInterval', 'Promo_StoreType']
    le = LabelEncoder()
    for feat in cat_features:
        df[feat] = le.fit_transform(df[feat].astype(str))
    
    # Rolling statistics (for train data only, as test data is for future prediction)
    if 'Sales' in df.columns:
        for window in [7, 14, 30]:
            df[f'Sales_Rolling_Mean_{window}'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'Sales_Rolling_Std_{window}'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(window, min_periods=1).std())
    
    return df

def main():
    train, test = load_and_preprocess_data('train.csv', 'test.csv', 'store.csv')
    
    train = engineer_features(train)
    test = engineer_features(test)
    
    # Save processed data
    train.to_csv('processed_train.csv', index=False)
    test.to_csv('processed_test.csv', index=False)
    
    print("Data preprocessing and feature engineering completed.")

if __name__ == "__main__":
    main()