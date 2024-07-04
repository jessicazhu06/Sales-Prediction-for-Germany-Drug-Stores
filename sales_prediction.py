#%%
#Data Manipulation and Treatment
import pandas as pd
import numpy as np
from datetime import datetime
#Plotting and Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
#Scikit-Learn for Modeling
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
#%%
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')
types = {
    'StateHoliday': str,
    'SchoolHoliday':str
}
df_train = pd.read_csv("train.csv", parse_dates=['Date'], date_parser=dateparse,dtype = types)
df_test = pd.read_csv("test.csv", parse_dates=['Date'], date_parser=dateparse,dtype = types)
df_store = pd.read_csv("store.csv")


# %%
# Find null values in train dataset
df_train.count(0)/df_train.shape[0] * 100
# the columns have got a good fill rate, so there is no null values in train dataset.
# We don't need to do any change for the train set for now at least.

#%%
df_test.count(0)/df_test.shape[0] * 100

# %%
# Data visualization of store, train and test dateset
import sweetviz as sv

store_report = sv.analyze(df_store)
train_report = sv.analyze(df_train)
test_report = sv.analyze(df_test)

# Generate a HTML report
store_report.show_html('Store_Sweetviz_Report.html')
train_report.show_html('Train_Sweetviz_Report.html')
test_report.show_html('Test_Sweetviz_Report.html')

#%%
# Plot histplot and kde of different variables in the dataset
def plot_histogram_and_boxplot(dataframe, column_name, ax):
    sns.histplot(data=dataframe, x=column_name, ax=ax)
    ax.set_title(f'{column_name} Histogram')

def plot_dataframe(unshow_feature, df):
    df = df.drop(columns=unshow_feature)
    total_len = len(df.columns)
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    for i, col in enumerate(df.columns):
        plot_histogram_and_boxplot(df, col, axes[i // 3, i % 3])

    plt.tight_layout()
    plt.show()

plot_dataframe('Store',df_store)
plot_dataframe('Store', df_train)
plot_dataframe('Store', df_test)
#%%
df_store.info()
# there are some missing values in store dataset
#%%
# deal with missing values
df_train.groupby('Date')['Store'].count().plot()
df_train.groupby('Store')['Date'].count().value_counts()
# there are two kinds of missing values, the first one only has 758 days during past years and the other one has 941 days during past years.
#print(df_train[df_train["Store"]==1].shape[0])
#%%
df_store.isnull().sum()
# Some findings:
# 1. There are sales = 0 in train dataset, and we need to discover more about this
# 2. There are some variables having missing values in Store dataset and Test dataset

#%%
#
df_store_CompetitionDistance_distribution=df_store.drop(df_store[pd.isnull(df_store.CompetitionDistance)].index)
# Box-plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  
plt.boxplot(df_store_CompetitionDistance_distribution.CompetitionDistance, showmeans=True,vert=False)
plt.title('Competition Boxplot')

# Histplot
plt.subplot(1, 2, 2)  
plt.hist(df_store_CompetitionDistance_distribution.CompetitionDistance, cumulative=False, bins=30)
plt.title("Competition histogram")
plt.xticks((min(df_store_CompetitionDistance_distribution.CompetitionDistance), max(df_store_CompetitionDistance_distribution.CompetitionDistance)))

plt.tight_layout() 
plt.show()

column_mean = df_store['CompetitionDistance'].mean()
column_median = df_store['CompetitionDistance'].median()
column_std = df_store['CompetitionDistance'].std()

print("Mean:", column_mean)
print("Median:", column_median)
print("Standard Deviation:", column_std)
# Mean of 'CompetitionDistance' is much larger than median of 'CompetitionDistance', because the mean is biased by the outliners, so we will use median to fill the null data for competitiondistance

#%%
# CompetitionDistance, filling the data using median
df_store[pd.isnull(df_store.CompetitionDistance)] 
df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace = True)
# CompetitionOpenSinceMonth and CompetitionOpenSinceYear, filling the data with 0
df_store.CompetitionOpenSinceMonth.fillna(0, inplace = True)
df_store.CompetitionOpenSinceYear.fillna(0,inplace=True)

#%%
# Check the null value in Store dataset
df_store.isnull().sum()

#%%
# Cheak the null of "Promo2SinceWeek, Promo2SinceYear and PromoInterval"
print("The number of rows that its Promo2SinceWeek is null is "+str(df_store[pd.isnull(df_store.Promo2SinceWeek)].shape[0]))
print("The number of rows that its Promo2SinceWeek is null and Promo2 is 0 is "+str(df_store[pd.isnull(df_store.Promo2SinceWeek)& (df_store.Promo2==0)].shape[0]))
print("The number of rows that its Promo2SinceYear is null and Promo2 is 0 is "+str(df_store[pd.isnull(df_store.Promo2SinceYear)& (df_store.Promo2==0)].shape[0]))
print("The number of rows that its PromoInterval is null and Promo2 is 0 is "+str(df_store[pd.isnull(df_store.PromoInterval)& (df_store.Promo2==0)].shape[0]))
# It means when there is no promotion, the other promotion variables will also be null, so fill null with 0 in these columns.

#%%
# Fill in null values with 0
df_store.Promo2SinceWeek.fillna(0, inplace = True)
df_store.Promo2SinceYear.fillna(0, inplace = True)
df_store.PromoInterval.fillna(0, inplace = True)

#%%
df_store.isna().sum()

#%%
# different reasons that stores are not open
print ()
open0 = df_train[(df_train.Open == 0)].count()[0]
print ("-Over those two years, {} is the number of times that different stores closed on given days.".format(open0))
print ()
open0_InSchoolHoliday = df_train[(df_train.Open == 0) & (df_train.SchoolHoliday == "1")&(df_train.StateHoliday == '0') ].count()[0]
print ("-From those closed events, {} times occured because there was a school holiday. " .format(open0_InSchoolHoliday))
print ()
open0_InStateHoliday = df_train[(df_train.Open == 0) &
         ((df_train.StateHoliday == 'a') |
          (df_train.StateHoliday == 'b') | 
          (df_train.StateHoliday == 'c'))].count()[0]
print ("-And {} times it occured because of either a bank holiday or easter or christmas.".format(open0_InStateHoliday))
print ()
open1_WithOutHoliday = df_train[(df_train.Open == 0) &
         (df_train.StateHoliday == "0")
         &(df_train.SchoolHoliday == "0")].count()[0]
print ("-But interestingly enough, {} times those shops closed on days for no apparent reason when no holiday was announced. In fact, those closings were done with no pattern whatsoever and in this case from 2013 to 2015 at almost any month and any day.".format(open1_WithOutHoliday))
print ()
#%%
# Drop train data that Sales > 0 and Open = 0
df_train = df_train[(df_train['Open'] != 0) & (df_train['Sales']>0)].copy()
df_train = df_train.reset_index(drop=True)
print ("New training set has {} rows now".format(df_train.shape[0]))

#%%
# fill open with 1, assume that they all open these days
df_test.Open.fillna(1,inplace = True)
# merge train dataset and test dataset with store dataset
train = pd.merge(df_train, df_store, how = 'left', on = 'Store')
test = pd.merge(df_test, df_store, how = 'left', on = 'Store')
train.to_csv("train_store.csv")

#%%
# check whether there are outliers in the store dataset
# Set up the matplotlib figure: 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)

# Plot a boxplot for Sales by StoreType
sns.boxplot(x='StoreType', y='Sales', data=train, ax=axes[0],color = "purple")
axes[0].set_title('Sales by StoreType')

# Plot a boxplot for Customers by StoreType
sns.boxplot(x='StoreType', y='Customers', data=train, ax=axes[1], color = "purple")
axes[1].set_title('Customers by StoreType')

# Display the plot
plt.tight_layout()
plt.show()
#there are two outliers in each dataset, storenums are 817 and 909

#%%
# drop the outliers
# find the indices of the maximum of 'sales' and 'customers', and drop them
sales_max = train['Sales'].max()
customers_max = train['Customers'].max()

indices_of_sales_max = train[train['Sales'] == sales_max].index
indices_of_customers_max = train[train['Customers'] == customers_max].index

train = train.drop(indices_of_sales_max)
train = train.drop(indices_of_customers_max)

#%%
#%%
'''
df_store['PromoInterval'] = df_store['PromoInterval'].astype(str)
plot_dataframe('Store',df_store)
'''

#%%
'''
plot_dataframe('Store', df_train)
'''
#%%
'''
df = df_store[df_store["CompetitionOpenSinceYear"]!=0]
plt.hist(df["CompetitionOpenSinceYear"])
plt.title("CompetitionOpenSinceYear")
'''
#%%
'''
df = df_store[df_store["Promo2SinceYear"]!=0]
plt.hist(df["Promo2SinceYear"])
plt.title("Promo2SinceYear")
'''

#%%
#Plot after data washing
train['PromoInterval'] = train['PromoInterval'].astype(str)
cleaned_train_report = sv.analyze(train)

# Generate a HTML report
cleaned_train_report.show_html('Traincleaned_Sweetviz_Report.html')

# %%
# Group data by StoreType
grouped = train.groupby('StoreType').agg(
    Total_Stores=('Store', 'nunique'),
    Total_Sales=('Sales', 'sum'),
    Total_Customers=('Customers', 'sum'),
    Average_Sales=('Sales', 'mean'),
    Average_Customers=('Customers', 'mean')
)
# Calculate average sales per customer
grouped['Avg_Sales_Per_Customer'] = grouped['Total_Sales'] / grouped['Total_Customers']

# Creating the subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
fig.suptitle('Statistics by Store Type', fontsize=16)

# Plotting each statistic
stats = ['Total_Stores', 'Total_Sales', 'Total_Customers', 'Average_Sales', 'Average_Customers', 'Avg_Sales_Per_Customer']
titles = ['Total Stores', 'Total Sales', 'Total Customers', 'Average Sales', 'Average Customers', 'Avg. Sales per Customer']

for ax, stat, title in zip(axes.flatten(), stats, titles):
    grouped[stat].plot(kind='bar', ax=ax, color="purple")
    ax.set_title(title)
    ax.set_xlabel('Store Type')
    ax.set_ylabel(stat.replace('_', ' '))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()
#%%
# Plot a boxplot for CompetitionDistance by StoreType
plt.figure(figsize=(10, 6))
sns.boxplot(x='StoreType', y='CompetitionDistance', data=train)
plt.title('Competition Distance by StoreType')
plt.show()

# %%
# Ensure that the 'Date' column is a datetime type and extract the year
train['Year'] = train['Date'].dt.year

# Create a line plot for the sales over years for different StoreTypes and Promo
plt.figure(figsize=(14, 7))

# Use sns.lineplot to aggregate the data and plot the mean sales for each year, StoreType, and Promo
sns.lineplot(x='Year', y='Sales', hue='StoreType', style='Promo', data=train, markers=True, dashes=False)

plt.title('Yearly Sales Trends by StoreType and Promo')
plt.xlabel('Year')
plt.ylabel('Average Sales')
plt.legend(title='StoreType/Promo')
plt.show()

#%%

# Filter out days when the store was closed (Sales would be 0 or store was not open)
data_open = train[train['Open'] == 1]

# For StateHoliday, make sure to consider '0' and 0 as non-holiday
data_open['StateHoliday'] = data_open['StateHoliday'].replace(0, '0')

# Group data by StoreType, Year, and StateHoliday, then calculate mean sales
grouped_sales = data_open.groupby(['StoreType', 'Year', 'StateHoliday']).agg(Average_Sales=('Sales', 'mean')).reset_index()

# Unique StoreTypes and Years for plotting
store_types = grouped_sales['StoreType'].unique()
years = sorted(grouped_sales['Year'].unique())

# Plotting
for store_type in store_types:
    fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(20, 5), sharey=True)
    fig.suptitle(f'Store Type {store_type} - Average Sales by State Holiday and Year', fontsize=16)
    
    for ax, year in zip(axes, years):
        df_subset = grouped_sales[(grouped_sales['StoreType'] == store_type) & (grouped_sales['Year'] == year)]
        if not df_subset.empty:
            df_subset.pivot(index='StateHoliday', columns='Year', values='Average_Sales').plot(kind='bar', ax=ax)
            ax.set_title(f'Year {year}')
            ax.set_xlabel('State Holiday')
            ax.set_ylabel('Average Sales')
        else:
            ax.set_title(f'Year {year} (No Data)')
            ax.set_xlabel('State Holiday')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols

# To avoid any confusion in subsequent analysis, unify the representation of StateHoliday column values to string '0'
train['StateHoliday'] = train['StateHoliday'].astype(str).replace('0.0', '0')

# List of categorical and continuous variables might need adjustment based on the actual columns in your data
categorical_vars = ['DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'Promo2']
continuous_vars = ['Sales', 'Customers','CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

# Analyzing the correlation between categorical variables and Sales (ANOVA)
print("ANOVA results for Categorical Variables:")
for var in categorical_vars:
    # Use C() to indicate categorical variables
    formula = f'Sales ~ C({var})'
    model = ols(formula, data=train).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
    print(f"\n{var}:")
    print(anova_results)
#%%
# Calculate the Pearson correlation coefficients for continuous variables
print("\nPearson Correlation Coefficients for Continuous Variables:")
df_correlation=train[continuous_vars]
fig, ax = plt.subplots(figsize = (15, 10))
sns.heatmap(df_correlation.corr(),ax=ax,fmt='.2f',annot=True,linewidths=0.5,cmap=sns.diverging_palette(10, 133, as_cmap=True))
plt.title('Correlation Matrix with Continuous Variables')

# %%
# Featuer Engineering and Feature Selection
from sklearn.preprocessing import LabelEncoder
def extract_features(features, data):
    
    # Use some properties directly
    direct_features = ['Store', 'CompetitionDistance']
    features.extend(direct_features)
    
    # Extract date features
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek + 1  # +1 to make Monday=1, Sunday=7
    data['WeekOfYear'] = data.Date.dt.isocalendar().week
    date_features = ['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear']
    features.extend(date_features)
    
    # Competition and Promo Open features
    data['CompetitionOpen'] = np.where(data['CompetitionOpenSinceYear'] == 0, 
                                   0, 
                                   12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['Month'] - data['CompetitionOpenSinceMonth']))
    
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    features.extend(['CompetitionOpen', 'PromoOpen'])
    
    data['PromoInterval'].fillna('0', inplace=True)
    # Label encode some features
    categorical_features = ['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 'Promo', 'Promo2']
    features.extend(categorical_features)
    label_encoders = {}
    for column in data[categorical_features]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    PromoInterval_maps = {'0':1, 'Jan,Apr,Jul,Oct':2, 'Feb,May,Aug,Nov':3, 'Mar,Jun,Sept,Dec':4}
    data.PromoInterval.replace(PromoInterval_maps, inplace=True)
    features.extend(['PromoInterval'])

    # Set as categories
    #feature_to_cate = ['Promo','SchoolHoliday','StoreType', 'Assortment', 'StateHoliday']
    #data[feature_to_cate] = data[feature_to_cate].astype('category')

    return features, data


# %%
features = []
train_features, train_preprocessed = extract_features(features, train)
test_features, test_preprocessed = extract_features([], test)
train_preprocessed = train[train_features]
test_preprocessed = test[test_features]

#%%
from sklearn.metrics import make_scorer
'''
def rmspe(y_true, y_pred):
    # Avoid division by zero
    mask = y_true != 0
    rmspe_val = np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))
    return rmspe_val
'''
def rmspe(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(np.mean(((y_true - y_pred) / y_true)**2))
                           
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# %%

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

'''
train_preprocessed = pd.read_csv("train_preprocessed")
train = pd.read_csv("train_cleaned")
'''

X = train_preprocessed
y = np.log1p(train['Sales'])


#%%
'''
# Using RandomizedSearchCV to tune parameters
xgbr = xgb.XGBRegressor(seed = 20)

params = { 'max_depth': [ 3, 5, 6, 10],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.5, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 300, 500]}


# Use GridSearchCV to tune the hyper-parameters
clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring=rmspe_scorer,  # Use RMSPE scorer here,
                         n_iter=25,
                         cv = 10,
                         verbose=1)
clf.fit(X, y)

# Round best parameters
rounded_best_params_clf = {key: round(value, 2) for key, value in clf.best_params_.items()}

# print the best parameters and model
print("Rounded Best Parameters: ", rounded_best_params_clf)
best_score = np.sqrt(-clf.best_score_)
print("Best RMSPE Score:", best_score)
'''

#%%
#Round 1
params = { 
         'max_depth': np.arange(3,11,1),
          'min_child_weight': np.arange(0.8,1.3,0.1)}
xgbr_1 = xgb.XGBRegressor(seed = 20,
                        learning_rate = 0.1, 
                        n_estimators = 300
)

# GridSearchCV to tune the hyper-parameters
clf_1 = GridSearchCV(estimator=xgbr_1,
                         param_grid=params,
                         scoring=rmspe_scorer,
                         cv = 10)
clf_1.fit(X, y)

#%%
# Round best parameters
rounded_best_params_clf_1 = {key: round(value, 2) for key, value in clf_1.best_params_.items()}

print("Rounded Best Parameters: ", rounded_best_params_clf_1)
best_score = -clf_1.best_score_
print("Best RMSPE Score:", best_score)
#%%
# Using step-wise and GridSearchCV to tune parameters
# Round 2
params = {
         'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.5, 1.0, 0.1)}
xgbr_2 = xgb.XGBRegressor(seed = 20,
                          learning_rate = 0.1, 
                          n_estimators = 300,
                          max_depth = 10, 
                          min_child_weight = 1.1)

# Use GridSearchCV to tune the hyper-parameters
clf_2 = GridSearchCV(estimator=xgbr_2,
                         param_grid=params,
                         scoring=rmspe_scorer,
                         cv = 10)
clf_2.fit(X, y)

# Round best parameters
rounded_best_params_clf_2 = {key: round(value, 2) for key, value in clf_2.best_params_.items()}

# %%
# print the best parameters and model
print("Rounded Best Parameters: ", rounded_best_params_clf_2)
best_score = -clf_2.best_score_
print("Best RMSPE Score:", best_score)
# %%
# Round 3
params = {'learning_rate': [0.01, 0.1, 0.2, 0.3],
         'n_estimators': [50, 100, 300, 500]}
xgbr_3 = xgb.XGBRegressor(seed = 20,
                          max_depth = 10, 
                          min_child_weight = 1.1,
                          subsample = 0.5,
                          colsample_bytree = 0.8)

# Use GridSearchCV to tune the hyper-parameters
clf_3 = GridSearchCV(estimator=xgbr_3,
                         param_grid=params,
                         scoring=rmspe_scorer,
                         cv = 10)
clf_3.fit(X, y)
# %%
# Round best parameters
rounded_best_params_clf_3 = {key: round(value, 2) for key, value in clf_3.best_params_.items()}

# print the best parameters and model
print("Rounded Best Parameters: ", rounded_best_params_clf_3)
best_score = -clf_3.best_score_
print("Best RMSPE Score:", best_score)

# %%
from xgboost import plot_importance
best_model = clf_3.best_estimator_
plot_importance(best_model, importance_type = 'weight', title = 'Feature Importances for XGBoost')
plt.show()
# %%

# Assuming you have your test dataset as X_test
# Predict using the best estimator from GridSearchCV
y_pred = clf_3.predict(test_preprocessed)

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame(np.expm1(y_pred), columns=['Predicted'])

predictions_df['Store'] = test_preprocessed['Store'].values
predictions_df['Date'] = test['Date'].values

predictions_df = predictions_df[['Store', 'Date', 'Predicted']]
# Optionally, if you have an identifier for your test set rows, add it to the DataFrame
# predictions_df['ID'] = test_ids  # Uncomment and replace test_ids with your actual IDs

# Save the DataFrame to a CSV file
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")

# %%
