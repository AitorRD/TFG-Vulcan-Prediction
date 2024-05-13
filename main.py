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

def train_and_predict(model_name, data_file, split_type):
    model_config = load_model_config()
    model_info = next((model for model in model_config['models'] if model['name'] == model_name), None)

    if model_info:
        train_module = importlib.import_module(model_info['module'])
        predict_module = importlib.import_module(model_info['module'].replace('train', 'predict').replace('model', 'predict'))
        train_func = getattr(train_module, model_info['function'])
        evaluate_func = getattr(predict_module, 'evaluate_' + model_name.lower())
        
        params = model_info['params']
        X_train_list, X_test_list, y_train_list, y_test_list, X_imputed = divide_data(data_file, split_type)
        trained_model = train_func(X_train_list, y_train_list, **params)
        evaluate_func(trained_model, X_test_list, y_test_list, data_file, X_imputed, model_name)
    else:
        print(f"El modelo {model_name} no está configurado en model.yaml")

def main():
    process_data_mode = "DEFAULT" #OPTIONS DEFAULT , TSFRESH
    model_name = "DT"  # OPTIONS: KNN , RF , DT
    split_type = "KFOLD" # OPTIONS: KFOLD , TT
    data_file = "src/data/processed/dataframe.csv"

    #process_data(process_data_mode)
    train_and_predict(model_name, data_file, split_type)

if __name__ == "__main__":
    main()