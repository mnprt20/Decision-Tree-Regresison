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
