import subprocess
import yaml
import importlib

def load_model_config():
    with open('model.yaml', 'r') as file:
        return yaml.safe_load(file)

def process_data(process_data_mode):
    try:
        if process_data_mode == 'MANUALFEATURES':
            subprocess.run(['python', 'src/data/process_data.py'], check=True)
        elif process_data_mode == 'TSFRESH':
            subprocess.run(['python', 'src/tsfresh/feature_extractor.py'], check=True)
            subprocess.run(['python', 'src/tsfresh/feature_selector.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing data processing script: {e}")

def divide_data(data_file, split_type):
    if split_type == 'CROSSVAL':
        module = importlib.import_module('src.data.divide_data')
        return module.divide_data_kfold(data_file)
    elif split_type == 'HOLDOUT':
        module = importlib.import_module('src.data.divide_data')
        return module.divide_data_holdout(data_file)
    else:
        raise ValueError("Invalid split type. Supported types are 'CROSSVAL' or 'HOLDOUT'")

def train_and_predict(model_name, data_file, split_type, process_data_mode):
    model_config = load_model_config()
    model_info = next((model for model in model_config['models'] if model['name'] == model_name), None)

    if model_info:
        train_module = importlib.import_module(model_info['module'])
        train_func = getattr(train_module, model_info['function'])
        
        params = model_info['params']
        X_train_list, X_test_list, y_train_list, y_test_list, X_imputed = divide_data(data_file, split_type)
        trained_model = train_func(X_train_list, y_train_list, **params)

        predict_module = importlib.import_module('src.predict.predict')
        evaluate_func = getattr(predict_module, 'evaluate_model')
        evaluate_func(trained_model, X_test_list, y_test_list, data_file, X_imputed, model_name, split_type, process_data_mode)
    else:
        print(f"Model {model_name} is not configured in model.yaml")

def optimize_model(model_name):
    optimization_module = importlib.import_module('src.model.optimization')
    if model_name == 'DT':
        optimization_module.optimize_dt()
    elif model_name == 'RF':
        optimization_module.optimize_rf()
    elif model_name == 'ADABOOST':
        optimization_module.optimize_ab()
    elif model_name == 'GBOOST':
        optimization_module.optimize_gboost()
    else:
        print(f"No optimization function available for model {model_name}")

def main():
    user_input = input("Do you want to process the raw data? (y/n): ").strip().lower()
    if user_input == 'y':
        process_data_mode = input("Enter data processing mode (MANUALFEATURES/TSFRESH): ").strip().upper()
        print("-------- PROCESSING DATA --------")
        process_data(process_data_mode)
        print("-------- DATA SAVED --------")
    else:
        print("Process skipped")
        process_data_mode = input("Enter data mode that is already processed (MANUALFEATURES/TSFRESH): ").strip().upper()

    data_file_manual_train = "src/data/processed/dataframe.csv"
    data_file_tsfresh_train = "src/tsfresh/processed/tsfresh_dataframe.csv"

    if process_data_mode == 'MANUALFEATURES':
        data_file = data_file_manual_train 
    elif process_data_mode == 'TSFRESH':
        data_file = data_file_tsfresh_train
    else:
        raise ValueError("Invalid data processing mode. Supported modes are 'MANUALFEATURES' or 'TSFRESH'")

    user_input = input("Do you want to run the model optimization? (y/n): ").strip().lower()
    if user_input == 'y':
        model_name = input("Enter the model you want to optimize (KNN/DT/RF/ADABOOST/GBOOST): ").strip().upper()
        print("-------- OPTIMIZING PARAMETERS --------")  
        optimize_model(model_name)
    else:
        print("Optimization skipped")
    
    split_type = input("Enter the split type you want to use (CROSSVAL/HOLDOUT): ").strip().upper()
    model_name = input("Enter the model you want to use (KNN/DT/RF/ADABOOST/GBOOST): ").strip().upper()
    print("-------- TRAINING MODEL --------")
    train_and_predict(model_name, data_file, split_type, process_data_mode)


if __name__ == "__main__":
    main()