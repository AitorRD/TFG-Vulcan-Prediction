from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn import feature_selection, feature_extraction #TODO hacer esto tmb 
import pandas as pd
import os
import time

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, directory, use_relevant_features=False):
        start_time = time.time()
        extracted_features = []

        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                time_series_data = pd.read_csv(filepath) 
                time_series_data['id'] = range(1, len(time_series_data) + 1)
                time_series_data = self.handle_nan_values(time_series_data)
                if use_relevant_features:
                    relevant_features = extract_relevant_features(time_series_data, column_id='id')
                    extracted_features.append(relevant_features)
                else:
                    features = extract_features(time_series_data, column_id='id', default_fc_parameters=MinimalFCParameters())
                    extracted_features.append(features)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Extracción de características completada en {execution_time} segundos")

        if use_relevant_features:
            filename = "tsfresh_processed_relevant_feat.csv"
        else:
            filename = "tsfresh_processed_minfc_feat.csv"

        output_dir = "src/data/processed"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df = pd.concat(extracted_features)
        df.to_csv(filepath, index=False)
        print(f"Características guardadas en {filepath}")

        return extracted_features
    
    def handle_nan_values(self, time_series_data):
        # Eliminar columnas completas con valores NaN
        time_series_data.dropna(axis=1, how='all', inplace=True)

        # Calcular la media de cada columna
        column_means = time_series_data.mean(axis=0)

        # Imputar la media a los valores NaN de cada columna
        time_series_data.fillna(column_means, inplace=True)

        return time_series_data

if __name__ == '__main__':
    feature_extractor = FeatureExtractor()
    directory = "../data/kaggle/input/train"
    extracted_features = feature_extractor.extract_features(directory, False)

#TODO MLP (Neuronal), ADABOOST y XGBoost mirarmelo
