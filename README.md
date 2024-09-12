# Decision-Tree-Regresison
**Weather Forecasting using Decision Tree Regressor**
=====================================================

### Overview
This project uses a Decision Tree Regressor to predict the air temperature at 9am based on various weather attributes. The dataset used is the daily weather dataset, which contains 1096 samples and 10 features.

### Dataset
The dataset is stored in a CSV file named "daily_weather.csv" and is located at "C:\\infosys\\archive (1)\\daily_weather.csv". The dataset contains the following features:

* **air_pressure_9am**
* **air_temp_9am** (target attribute)
* **avg_wind_direction_9am**
* **avg_wind_speed_9am**
* **max_wind_direction_9am**
* **max_wind_speed_9am**
* **rain_accumulation_9am**
* **rain_duration_9am**
* **relative_humidity_9am**
* **relative_humidity_3pm**

### Code Explanation
#### Importing Libraries
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
```
### Data Loading and Exploration
```python
df = pd.read_csv("C:\\infosys\\archive (1)\\daily_weather.csv")
print(df)
```
### Correlation Matrix
```python
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
```
### Pairplot
```python
sns.pairplot(df, vars=['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am', 'avg_wind_speed_9am', 
                       'max_wind_direction_9am', 'max_wind_speed_9am', 'rain_accumulation_9am', 
                       'rain_duration_9am', 'relative_humidity_9am', 'relative_humidity_3pm'])
plt.show()
```
### Handling Missing Values
```python
print(df.isnull().values.any())
df.dropna(inplace=True)
print(df.isnull().values.any())
``` 
### Feature and Target Attribute Selection
```python
target_attribute = 'air_temp_9am'
X = df.drop(columns=[target_attribute])
y = df[target_attribute]
```
### Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Decision Tree Regressor
```python
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
y_pred = dt_regressor.predict(X_test)
```
### Evaluation Metrics
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
```
### Visualization
```python
sns.pairplot(df, x_vars=X.columns, y_vars=[target_attribute], height=3, aspect=1)
plt.show()

plt.figure(figsize=(10,8))
plot_tree(dt_regressor, feature_names=X.columns, filled=True)
plt.show()

treemodel = DecisionTreeRegressor()
treemodel.fit(X_train, y_train)

plt.figure(figsize=(15,10))
tree.plot_tree(treemodel, filled=True)
plt.show()
```
### Requirements
* **Python 3.x**
* **pandas**
* **seaborn**
* **matplotlib**
* **scikit-learn**

### Running the Code
To run the code, simply execute the Python script in a suitable environment. The code will load the dataset, perform the necessary data exploration and visualization, train the Decision Tree Regressor, and evaluate its performance.
