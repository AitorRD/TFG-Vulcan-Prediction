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
    module = importlib.import_module('data.divide_data')
    return module.divide_data(data_file)

def train_and_predict(model_name, data_file):
    model_config = load_model_config()
    model_info = next((model for model in model_config['models'] if model['name'] == model_name), None)
    if model_info:
        module = importlib.import_module('model.' + model_info['module'])
        train_and_predict_func = getattr(module, model_info['function'])
        X_train, X_test, y_train, y_test = divide_data(data_file)
        train_and_predict_func(X_train, X_test, y_train, y_test)
    else:
        print(f"El modelo {model_name} no está configurado en model.yaml")

def main():
    process_data()
    model_name = "KNN"  # OPCIONES: KNN , RF , DT
    data_file = "../data/processed/dataframe2.csv"
    train_and_predict(model_name, data_file)

if __name__ == "__main__":
    main()