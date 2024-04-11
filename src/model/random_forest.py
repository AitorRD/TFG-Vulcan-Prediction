import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
df = pd.read_csv('../data/dataframe2.csv')

X = df.drop(columns=['time_to_eruption'])
y = df['time_to_eruption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=50, random_state=50)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mape_rf = calculate_mape(y_test, y_pred_rf)

print("Random Forest:")
print("MAE:", mae_rf)
print("MSE:", mse_rf)
print("MAPE:", mape_rf)