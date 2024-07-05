import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def rmspe(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true)**2))

rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

def train_model(X, y):
    params = {
        'max_depth': [10],
        'min_child_weight': [1.1],
        'subsample': [0.5],
        'colsample_bytree': [0.8],
        'learning_rate': [0.1],
        'n_estimators': [500]
    }
    
    xgbr = xgb.XGBRegressor(seed=20)
    clf = GridSearchCV(estimator=xgbr, param_grid=params, scoring=rmspe_scorer, cv=10)
    clf.fit(X, y)
    
    return clf.best_estimator_

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    rmspe_value = rmspe(y, y_pred)
    return rmspe_value

def plot_feature_importance(model, features):
    import matplotlib.pyplot as plt
    
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(pos, feature_importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(features)[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for XGBoost Model')
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png')
    plt.close()

if __name__ == "__main__":
    train = pd.read_csv("../data/train_processed.csv")
    test = pd.read_csv("../data/test_processed.csv")
    
    features = [col for col in train.columns if col not in ['Date', 'Sales', 'Customers']]
    X = train[features]
    y = np.log1p(train['Sales'])
    
    model = train_model(X, y)
    rmspe_value = evaluate_model(model, X, y)
    print(f"Model RMSPE: {rmspe_value}")
    
    plot_feature_importance(model, features)
    
    test_predictions = np.expm1(model.predict(test[features]))
    pd.DataFrame({
        'Store': test['Store'],
        'Date': test['Date'],
        'Sales': test_predictions
    }).to_csv("../results/predictions.csv", index=False)
    
    print("Modeling and evaluation complete. Results saved.")