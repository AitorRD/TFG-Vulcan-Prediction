import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(model, X_test_list, y_test_list, data_file, X_imputed, model_name, divide_data_name, process_data_mode):
    df = pd.read_csv(data_file)
    y_pred_list = []
    mse_scores = []
    mae_scores = []
    mape_scores = []
    
    for X_test, y_test in zip(X_test_list, y_test_list):
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)
    
    y_pred = np.concatenate(y_pred_list)
    y_pred_to_save = model.predict(X_imputed)
    res = print_save_results(df, mae_scores, mse_scores, mape_scores, y_pred_to_save, model_name, divide_data_name, process_data_mode)
    return res

def print_save_results(df, mae_scores, mse_scores, mape_scores, y_pred_to_save, modelname, divide_data_name, process_data_mode):
    average_mae = sum(mae_scores) / len(mae_scores)
    average_mse = sum(mse_scores) / len(mse_scores)
    average_mape = sum(mape_scores) / len(mape_scores)
    
    print(f"{modelname} -> Error absoluto medio (MAE):", average_mae)
    print(f"{modelname} -> Error cuadrático medio (MSE):", average_mse)
    print(f"{modelname} -> Error porcentaje absoluto medio (MAPE):", average_mape)
    
    results_df = pd.DataFrame({"volcan_id": df['volcan_id'], "time_to_eruption_dt": y_pred_to_save})
    results_df.to_csv(f"src/predict/results/results_{modelname.lower()}_{divide_data_name.lower()}_{process_data_mode.lower()}.csv", index=False)
    
    print('Saved Results')
