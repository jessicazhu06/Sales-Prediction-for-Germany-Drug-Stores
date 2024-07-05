# Sales Prediction Project

## Overview
This project aims to predict sales for ROSSMANN, a large drug store chain in Germany. Using historical sales data from 1,115 stores, we developed a model to forecast sales for the next 6 weeks.

## Data
The dataset used in this project is available publicly on Kaggle: https://www.kaggle.com/competitions/rossmann-store-sales/data

The dataset includes:
- Store information (1,115 stores with 10 variables)
- Historical sales data (01/01/2013 to 31/07/2015)
- Test data for prediction (01/08/2015 to 17/09/2015)

## Methodology
1. Data Preparation and Cleaning
   - Handled missing values in store dataset
   - Removed outliers and invalid data points
   - Merged store information with sales data

2. Exploratory Data Analysis (EDA)
   - Analyzed sales patterns across different store types
   - Investigated the impact of promotions and holidays on sales
   - Examined the relationship between competition and sales

3. Feature Engineering
   - Extracted time-based features (year, month, day, day of week)
   - Created features for competition and promotion duration
   - Encoded categorical variables

4. Modeling
   - Used XGBoost Regressor
   - Performed hyperparameter tuning using GridSearchCV
   - Evaluated model using Root Mean Square Percentage Error (RMSPE)

## Key Findings
1. Store type 'b' has the highest average sales despite having the shortest distance to competitors.
2. Promotions have a significant positive impact on sales, especially for store type 'A'.
3. The most important features for predicting sales are:
   - Day of the month
   - Store
   - Competition duration
   - Competition distance
   - Day of the week

## Results
Our final model achieved an RMSPE of 0.1995 on the test set.

## Future Work
- Incorporate weather data for potentially improved predictions
- Experiment with ensemble methods combining multiple models
- Develop a more robust approach for handling stores with zero sales on open days

## Repository Structure
- `notebook/`: A jupyter notebook for data cleaning and exploratory data analysis
- `src/`: Python scripts for data processing and modeling
- `results/`: Output files and visualizations

## To run this project:
1. Download the dataset from Kaggle and place the CSV files in a `data/` directory in the root of this project.
2. Ensure you have the required Python packages installed: **pip install -r requirements.txt**
3. Run the preprocessing script: **src/data_preprocessing.py**
4. Run the modeling and evaluation script: **src/modeling_and_evaluation.py**
5. Alternatively, run the entire pipeline using: **main.py**
   
## Requirements
See `requirements.txt` for a list of required Python packages.
