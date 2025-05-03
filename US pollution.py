
# U.S Pollution Dataset
# Dataset scraped from the database of U.S. EPA
# Kaggle: https://www.kaggle.com/datasets/sogun3/uspollution/data

# Author: Jennifer Oviedo

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("pollution_us_2000_2016.csv")
print(f'First 5 records:\n {df.head()}')
print(f'Last 5 records:\n {df.tail()}')
print(f'Data shape: {df.shape}')    # Data shape: (1746661, 29)

print(df.info())

# Check missing values 
print(df.isnull().sum())

# Drop null values 
pollution_df = df.dropna().reset_index(drop=True)
print(pollution_df.isnull().sum())
print(f'Data shape after removing null values: {pollution_df.shape}') 
# Out: Data shape after removing null values: (436876, 29)

# Converting Date feature to date-time format
pollution_df['Date Local'] = pd.to_datetime(pollution_df['Date Local'])

# Creating 3 new features: Month, Year
pollution_df['Month'] = pollution_df['Date Local'].dt.month
pollution_df['Year'] = pollution_df['Date Local'].dt.year

# State Abreviations - better readibility
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Change state name with abbreviations
pollution_df['State'] = pollution_df['State'].map(state_abbrev)


# Drop variables
features = pollution_df.drop(columns=["Unnamed: 0","Address", "County",
                                      "Site Num","NO2 Units", "O3 Units",
                                      "SO2 Units", "CO Units"])
print(f'Features shape: {features.shape}')  # Output: (436876, 23)


# Bar Plots - Average Air Quality Index by State

# NO2 - Nitrogen dioxide 

NO2_state = features.groupby('State')['NO2 AQI'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
plt.bar(NO2_state.index, NO2_state.values)
plt.xlabel('State')
plt.ylabel('Average NO2 AQI')
plt.title('Average NO2 AQI by State')
plt.xticks(rotation=90)
plt.show()

# O3 - Ground-level ozone

O3_state = features.groupby('State')['O3 AQI'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
plt.bar(O3_state.index, O3_state.values)
plt.xlabel('State')
plt.ylabel('Average O3 AQI')
plt.title('Average O3 AQI by State')
plt.xticks(rotation=90)
plt.show()


# SO2 - Sulfur Dioxide

SO2_state = features.groupby('State')['SO2 AQI'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
plt.bar(SO2_state.index, SO2_state.values)
plt.xlabel('State')
plt.ylabel('Average SO2 AQI')
plt.title('Average SO2 AQI by State')
plt.xticks(rotation=90)
plt.show()


# CO - Carbon monoxide

CO_state = features.groupby('State')['CO AQI'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
plt.bar(CO_state.index, CO_state.values)
plt.xlabel('State')
plt.ylabel('Average CO AQI')
plt.title('Average CO AQI by State')
plt.xticks(rotation=90)
plt.show()


# Trend Over Time

trend_by_year = features.groupby('Year')[['NO2 AQI', 'O3 AQI', 
                                          'SO2 AQI', 'CO AQI']].mean() 

plt.figure(figsize=(12, 6))
plt.plot(trend_by_year.index, trend_by_year['NO2 AQI'], label = 'NO2 AQI', marker='o')
plt.plot(trend_by_year.index, trend_by_year['O3 AQI'], label = 'O3 AQI', marker ='o')
plt.plot(trend_by_year.index, trend_by_year['SO2 AQI'], label = 'SO2 AQI', marker='o')
plt.plot(trend_by_year.index, trend_by_year['CO AQI'], label = 'CO AQI', marker = 'o')
plt.xlabel('Year')
plt.ylabel('Average AQI of Each Pollutant Over Time')
plt.title('AQI over the years')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Feature Engineering - Overall AQI 
features['Overall_AQI'] = features[['NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI']].max(axis=1)


# Correlation Heatmap - 21 Variables

numeric_features = features.drop(columns=['State', 'City', 'State Code', 
                                          'County Code'])

sns.heatmap(numeric_features.corr(numeric_only=True), cmap='coolwarm')
plt.title('Correlation Heatmap of All Numeric Features')
plt.show()

# Heatmap shows strong collinearity between the pollutant mean and their 
# corresponing AQI values. Additionally, features that contain Max Hour values
# show little to no correlation. For these reasons, those features will be removed
# to improve the models. 

predictors = numeric_features[['NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI', 
                               'Year', 'Overall_AQI']]
sns.heatmap(predictors.corr(numeric_only=True), cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap of All Predictors')
plt.show()


# Feature Selection Lasso Regression 

predictors = numeric_features[['NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI', 
                               'Year', 'Overall_AQI']].copy()

lasso_x = predictors.drop(columns=['Overall_AQI'])
scaler = StandardScaler()
scaled = scaler.fit_transform(lasso_x)
lasso_y = numeric_features['Overall_AQI']
lasso = Lasso(alpha=1)
lasso_coeff = lasso.fit(scaled, lasso_y).coef_
plt.bar(lasso_x.columns, lasso_coeff)
plt.xticks(rotation = 45)
plt.title("Lasso Regression")
plt.show()


# Multiple Linear Regression Model for Overall AQI

X = predictors[['NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = predictors['Overall_AQI'] 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                    random_state= 42)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f'R2: {r2:.2f}')      # Out: R2: 0.88
rmse = mean_squared_error(y_test, y_pred)
print(f'MSE: {rmse:.2f}')   # Out: MSE: 44.14  


# Predicted vs. Actual 
start = min(y_test.min(), y_pred.min())
end = max(y_test.max(), y_pred.max())

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Overall AQI')
plt.ylabel('Predicted Overall AQI')
plt.title('Actual vs. Predicted values - Linear Regression')
plt.plot([start, end], [start, end], 'k--')
plt.show()

# Histogram - Residuals
residuals = y_test - y_pred
sns.histplot(residuals, bins=50, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Prediction Error")
plt.show()


# Classifying AQI in Categories (Levels of Concern)

index = [0, 50, 100, 150, 200, 300, 500]
quality = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 
           'Very Unhealthy', 'Hazardous']

features['AQI_Category'] = pd.cut(features['Overall_AQI'], bins = index, 
                                  labels = quality, right = True) 

features['AQI_Category'].value_counts()
# Out:
    # AQI_Category
    # Good                              368342
    # Moderate                           58499
    # Unhealthy for Sensitive Groups      9019
    # Unhealthy                            947
    # Very Unhealthy                        69
    # Hazardous                              0
    # Name: count, dtype: int64


    
# Neural Network

import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential

categories = {'Good': 0,
              'Moderate': 1,
              'Unhealthy for Sensitive Groups': 2,
              'Unhealthy': 3,
              'Very Unhealthy': 4}

predictors = features[['NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI', 'Year']]
predictors = scaler.fit_transform(predictors)
print(features.columns)

features['AQI_Category'] = pd.cut(features['Overall_AQI'], bins = index, 
                                  labels = quality, right = True) 
target = to_categorical(features['AQI_Category'].map(categories).values)
print(target)

ncols = predictors.shape[1]
print(ncols)

model = Sequential()
model.add(Input(shape=(ncols,)))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation = 'softmax'))
model.compile(optimizer="adam",
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(predictors, target, epochs=20, batch_size= 32)
print(model.summary())

# Classification Report - Neural Network

y_pred_classes = np.argmax(model.predict(predictors), axis=1)
y_true_classes = np.argmax(target, axis=1)

class_names = ['Good', 'Moderate', 
               'Unhealthy for Sensitive Groups', 
               'Unhealthy', 'Very Unhealthy']

print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))


# Random Forest - AQI Classification

categories = {'Good': 0,
              'Moderate': 1,
              'Unhealthy for Sensitive Groups': 2,
              'Unhealthy': 3,
              'Very Unhealthy': 4}
features['AQI_Category'] = pd.cut(features['Overall_AQI'], bins = index, 
                                  labels = quality, right = True) 

train_forest = features[features['Year'] <= 2010]
test_forest = features[features['Year'] > 2010]

X_f_train = train_forest[['State Code', 'County Code', 'NO2 AQI', 'O3 AQI', 
                          'SO2 AQI', 'CO AQI', 'Year']]
y_f_train = train_forest['AQI_Category']

X_f_test = test_forest[['State Code', 'County Code', 'NO2 AQI', 'O3 AQI', 
                        'SO2 AQI', 'CO AQI', 'Year']]
y_f_test = test_forest['AQI_Category']


rf = RandomForestClassifier(n_jobs=-1, random_state=42)
rf.fit(X_f_train, y_f_train)

y_f_pred = rf.predict(X_f_test)


print("Accuracy:", accuracy_score(y_f_test, y_f_pred))  
# Out: Accuracy: 0.999970352625868

print("\nClassification Report:\n", classification_report(y_f_test, y_f_pred))
# Out:
    # Classification Report:
    #                                  precision    recall  f1-score   support

    #                           Good       1.00      1.00      1.00    148510
    #                       Moderate       1.00      1.00      1.00     17618
    #                      Unhealthy       1.00      0.98      0.99       241
    # Unhealthy for Sensitive Groups       1.00      1.00      1.00      2276
    #                 Very Unhealthy       1.00      0.75      0.86         4

    #                       accuracy                           1.00    168649
    #                      macro avg       1.00      0.95      0.97    168649
    #                   weighted avg       1.00      1.00      1.00    168649


# Confusion Matrix

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_f_test, y_f_pred), annot=True, fmt='d',
            xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()



