import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_dt(tree_model, X_test, y_test, data_file, X_imputed):
    df = pd.read_csv(data_file)
    y_pred = tree_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    print("DT -> Error absoluto medio (MAE):", mae)
    print("DT -> Error cuadrático medio (MSE):", mse)
    print("DT -> Error porcentaje absoluto medio (MAPE):", mape)

    y_pred_to_save = tree_model.predict(X_imputed)
    results_df = pd.DataFrame({"volcan_id": df['volcan_id'], "time_to_eruption_dt": y_pred_to_save})
    results_df.to_csv("src/train/results/results_dt.csv", index=False)
    print('Saved Results')