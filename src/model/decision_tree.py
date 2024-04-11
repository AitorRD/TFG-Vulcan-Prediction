import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv('../data/dataframe2.csv')

X = df.drop(columns=['time_to_eruption', 'volcan_id'])
y = df['time_to_eruption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(random_state=50)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
mape_tree = calculate_mape(y_test, y_pred_tree)

print("Árbol de Decisión")
print("MAE:", mae_tree)
print("MSE:", mse_tree)
print("MAPE:", mape_tree)


