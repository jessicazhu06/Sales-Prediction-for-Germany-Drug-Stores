import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def load_data():
    train = pd.read_csv('processed_train.csv')
    test = pd.read_csv('processed_test.csv')
    return train, test

def prepare_features(df):
    features = [col for col in df.columns if col not in ['Date', 'Sales', 'Customers']]
    X = df[features]
    if 'Sales' in df.columns:
        y = np.log1p(df['Sales'])
    else:
        y = None
    return X, y

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))

def rmspe_xg(y_pred, y_true):
    y_true = y_true.get_label()
    return 'RMSPE', rmspe(y_true, y_pred)

def train_model(X_train, y_train):
    params = {
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 300, 500],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist')
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print("Best parameters:", grid_search.best_params_)
    return best_model

def main():
    train, test = load_data()
    
    X_train, y_train = prepare_features(train)
    X_test, _ = prepare_features(test)
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model = train_model(X_train_split, y_train_split)
    
    # Validate model
    val_pred = model.predict(X_val)
    val_rmspe = rmspe(np.expm1(y_val), np.expm1(val_pred))
    print(f"Validation RMSPE: {val_rmspe}")
    
    # Retrain on full dataset
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    test_pred = np.expm1(model.predict(X_test))
    
    # Prepare submission
    submission = pd.DataFrame({
        'Id': test['Id'],
        'Sales': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    
    print("Predictions completed and saved to submission.csv")

if __name__ == "__main__":
    main()