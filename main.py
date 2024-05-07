import subprocess
import yaml
import importlib

def load_model_config():
    with open('model.yaml', 'r') as file:
        return yaml.safe_load(file)

def process_data():
    try:
        subprocess.run(['python', 'src/data/process_data.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar process_data.py: {e}")

def divide_data(data_file):
    module = importlib.import_module('src.data.divide_data')
    return module.divide_data(data_file)

def train_and_predict(model_name, data_file):
    model_config = load_model_config()
    model_info = next((model for model in model_config['models'] if model['name'] == model_name), None)
    if model_info:
        module = importlib.import_module(model_info['module'])
        train_and_predict_func = getattr(module, model_info['function'])

        if 'params' in model_info:
            params = model_info['params']
            X_train_list, X_test_list, y_train_list, y_test_list, X_imputed = divide_data(data_file)
            trained_model = train_and_predict_func(X_train_list, y_train_list, **params)
            
            if model_name == "KNN":
                evaluate_knn_func = getattr(importlib.import_module('src.train.predict_knn'), 'evaluate_knn')
                evaluate_knn_func(trained_model, X_test_list, y_test_list, data_file, X_imputed)
            
            elif model_name == "DT":
                evaluate_dt_func = getattr(importlib.import_module('src.train.predict_dt'), 'evaluate_dt')
                evaluate_dt_func(trained_model, X_test_list, y_test_list, data_file, X_imputed)
            
            elif model_name == "RF":
                evaluate_dt_func = getattr(importlib.import_module('src.train.predict_rf'), 'evaluate_rf')
                evaluate_dt_func(trained_model, X_test_list, y_test_list, data_file, X_imputed)
        else:
            train_and_predict_func(data_file)
    else:
        print(f"El modelo {model_name} no está configurado en model.yaml")

def main():
    process_data()
    model_name = "RF"  # OPCIONES: KNN , RF , DT
    data_file = "src/data/processed/dataframe.csv"
    train_and_predict(model_name, data_file)

if __name__ == "__main__":
    main()