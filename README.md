# Sales Prediction Project

## Table of Contents
1. [Overview](#overview)
2. [Data](#data)
3. [Repository Structure](#repository-structure)
4. [Methodology](#methodology)
5. [Key Findings](#key-findings)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Running the Project](#to-run-this-project)
9. [Requirements](#requirements)

## Overview
This project aims to predict sales for ROSSMANN, a large drug store chain in Germany. Using historical sales data from 1,115 stores, we developed a model to forecast sales for the next 6 weeks.

## Data
The dataset used in this project is available publicly on Kaggle: https://www.kaggle.com/competitions/rossmann-store-sales/data

The dataset includes:
- Store information (1,115 stores with 10 variables)
- Historical sales data (01/01/2013 to 31/07/2015)
- Test data for prediction (01/08/2015 to 17/09/2015)

## Repository Structure
- `notebook/`: A jupyter notebook for data cleaning and exploratory data analysis
- `src/`: Python scripts for data processing and modelling
- `results/`: Output files and visualisations

## Methodology
1. Data Preparation and Cleaning
   - Handled missing values in store dataset
   - Removed outliers and invalid data points
   - Merged store information with sales data

2. Exploratory Data Analysis (EDA)
   - Analysed sales patterns across different store types
   - Investigated the impact of promotions and holidays on sales
   - Examined the relationship between competition and sales

3. Feature Engineering
   - Extracted time-based features (year, month, day, day of week)
   - Created features for competition and promotion duration
   - Encoded categorical variables

4. Modelling
   - Used XGBoost Regressor
   - Performed hyperparameter tuning using GridSearchCV
   - Evaluated model on test data using Root Mean Square Percentage Error (RMSPE)

## Key Findings
1. Store Performance:
   - Store type 'b' has the highest average sales despite having the shortest distance to competitors, suggesting a strong product differentiation or optimal location strategy.
   - Store type 'a' shows the highest sensitivity to promotions, indicating a more price-sensitive customer base.
     
2. Temporal Patterns:
   - Sales show strong weekly patterns with peaks on weekends and troughs mid-week, reflecting typical shopping behavior.
   - Yearly seasonality is evident, with sales peaks during holiday seasons (Christmas, Easter) and summer months.

3. Promotion Impact:
   - Promotions have a significant positive impact on sales, with an average increase of 18.63% across all store types.
   - The effectiveness of promotions varies by store type and day of the week, suggesting the need for targeted promotional strategies.

4. Competition Effects:
   - Stores with nearby competitors (within 1km) show 6.79% lower average sales, highlighting the importance of location strategy.
   - However, the impact of competition diminishes over time, possibly due to market segmentation or improved competitive strategies.

## Results
- Model Performance: Our final XGBoost model achieved an RMSPE of 0.0995 (9.95%) on the test set, indicating strong predictive power.

- Feature Insights: SHAP analysis revealed that recent sales history, promotion status, and store-specific factors are the strongest predictors of future sales.

- Actionable Insights:
   - Stores should focus on product differentiation to mitigate competition effects.
   - Promotional strategies should be tailored by store type and day of the week for maximum impact.
   - Inventory management should account for both short-term (weekly) and long-term (seasonal) sales patterns identified by the model.

## Future Work
- Incorporate weather data for potentially improved predictions
- Experiment with ensemble methods combining multiple models
- Develop a more robust approach for handling stores with zero sales on open days
  
## To run this project:
1. Download the dataset from Kaggle and place the CSV files in a `data/` directory in the root of this project.
2. Ensure you have the required Python packages installed: **pip install -r requirements.txt**
3. Run the preprocessing script: **src/data_preprocessing.py**
4. Run the modeling and evaluation script: **src/modeling_and_evaluation.py**
5. Alternatively, run the entire pipeline using: **main.py**
   
## Requirements
See `requirements.txt` for a list of required Python packages.
