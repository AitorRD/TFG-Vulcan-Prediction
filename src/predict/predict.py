import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_rf(rf_model, X_test_list, y_test_list, data_file, X_imputed, modelname):
    df = pd.read_csv(data_file)
    y_pred_list = []
    mse_scores = []
    mae_scores = []
    mape_scores = []
    
    for X_test, y_test in zip(X_test_list, y_test_list):
        y_pred = rf_model.predict(X_test)
        y_pred_list.append(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)

    y_pred = np.concatenate(y_pred_list)
    y_pred_to_save = rf_model.predict(X_imputed)
    res = print_save_results(df, mae_scores, mse_scores, mape_scores, y_pred_to_save, modelname)
    return res

def evaluate_dt(tree_model, X_test_list, y_test_list, data_file, X_imputed, modelname):
    df = pd.read_csv(data_file)
    y_pred_list = []
    mse_scores = []
    mae_scores = []
    mape_scores = []
    
    for X_test, y_test in zip(X_test_list, y_test_list):
        y_pred = tree_model.predict(X_test)
        y_pred_list.append(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)

    y_pred = np.concatenate(y_pred_list)
    y_pred_to_save = tree_model.predict(X_imputed)
    res = print_save_results(df, mae_scores, mse_scores, mape_scores, y_pred_to_save, modelname)
    return res

def evaluate_knn(knn, X_test_list, y_test_list, data_file, X_imputed, modelname):
    df = pd.read_csv(data_file)
    y_pred_list = []
    mse_scores = []
    mae_scores = []
    mape_scores = []
    
    for X_test, y_test in zip(X_test_list, y_test_list):
        y_pred = knn.predict(X_test)
        y_pred_list.append(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)
    
    y_pred = np.concatenate(y_pred_list)
    y_pred_to_save = knn.predict(X_imputed)
    res = print_save_results(df, mae_scores, mse_scores, mape_scores, y_pred_to_save, modelname)
    return res
     
def print_save_results(df, mae_scores, mse_scores, mape_scores, y_pred_to_save, modelname):
    average_mae = sum(mae_scores) / len(mae_scores)
    average_mse = sum(mse_scores) / len(mse_scores)
    average_mape = sum(mape_scores) / len(mape_scores)
    
    print(f"{modelname} -> Error absoluto medio (MAE):", average_mae)
    print(f"{modelname} -> Error cuadrático medio (MSE):", average_mse)
    print(f"{modelname} -> Error porcentaje absoluto medio (MAPE):", average_mape)
    
    results_df = pd.DataFrame({"volcan_id": df['volcan_id'], "time_to_eruption_dt": y_pred_to_save})
    results_df.to_csv(f"src/predict/results/results_{modelname.lower()}.csv", index=False)
    
    print('Saved Results')
