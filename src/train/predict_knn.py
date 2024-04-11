import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_knn(knn, X_test, y_test):
    y_pred = knn.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    print("KNN -> Error absoluto medio (MAE):", mae)
    print("KNN -> Error cuadrático medio (MSE):", mse)
    print("KNN -> Error porcentaje absoluto medio (MAPE):", mape)

    return y_pred

def save_results(segment_ids, predictions):
    results_df = pd.DataFrame({"segment_id": segment_ids, "time_to_eruption_pred": predictions})
    results_df.to_csv("results/results_knn.csv", index=False)