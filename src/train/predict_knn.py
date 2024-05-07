import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_knn(knn, X_test, y_test, data_file, X_imputed):
    df = pd.read_csv(data_file)
    y_pred = knn.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    print("KNN -> Error absoluto medio (MAE):", mae)
    print("KNN -> Error cuadrático medio (MSE):", mse)
    print("KNN -> Error porcentaje absoluto medio (MAPE):", mape)

    y_pred_to_save = knn.predict(X_imputed)
    results_df = pd.DataFrame({"volcan_id": df['volcan_id'], "time_to_eruption_knn": y_pred_to_save})
    results_df.to_csv("src/train/results/results_knn.csv", index=False)
    print('Saved Results')