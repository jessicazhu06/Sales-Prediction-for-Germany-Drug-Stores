import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, clean_data, extract_features
from src.modeling_and_evaluation import train_model, evaluate_model, plot_feature_importance

def main():
    # Data Preprocessing
    train, test, store = load_data("data/train.csv", "data/test.csv", "data/store.csv")
    train_cleaned, test_cleaned = clean_data(train, test, store)
    train_processed, features = extract_features(train_cleaned)
    test_processed, _ = extract_features(test_cleaned)
    
    # Save processed data
    train_processed.to_csv("data/train_processed.csv", index=False)
    test_processed.to_csv("data/test_processed.csv", index=False)
    print("Data preprocessing complete. Processed files saved.")
    
    # Modeling and Evaluation
    X = train_processed[features]
    y = np.log1p(train_processed['Sales'])
    
    model = train_model(X, y)
    rmspe_value = evaluate_model(model, X, y)
    print(f"Model RMSPE: {rmspe_value}")
    
    plot_feature_importance(model, features)
    
    test_predictions = np.expm1(model.predict(test_processed[features]))
    pd.DataFrame({
        'Store': test_processed['Store'],
        'Date': test_processed['Date'],
        'Sales': test_predictions
    }).to_csv("results/predictions.csv", index=False)
    
    print("Modeling and evaluation complete. Results saved.")

if __name__ == "__main__":
    main()