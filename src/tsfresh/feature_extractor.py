from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn import feature_selection, feature_extraction #TODO hacer esto tmb 
import pandas as pd
import os
import time

class FeatureExtractor:
    def __init__(self):
        self.counter = 1

    def extract_features(self, directory):
        start_time = time.time()
        extracted_features = []

        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory, filename)
                time_series_data = pd.read_csv(filepath) 
                time_series_data.insert(0, 'id', self.counter)
                self.counter += 1
                time_series_data.insert(1, 'time', range(1, len(time_series_data) + 1))
                time_series_data = self.procesar_sensores(time_series_data)
                
                features = extract_features(time_series_data, column_id='id', column_sort='time', default_fc_parameters=MinimalFCParameters())
                impute(features)

                volcan_id = os.path.splitext(filename)[0]
                features.insert(0, 'volcan_id', volcan_id)
                extracted_features.append(features)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Extracción de características completada en {execution_time/60} minutos")

        filename = "tsfresh_data_minfc.csv"
        output_dir = "processed"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df = pd.concat(extracted_features)
        df.to_csv(filepath, index=False)
        print(f"Características guardadas en {filepath}")

        return extracted_features
    
    def procesar_sensores(self, time_series_data):
        time_series_data['media_sensor'] = time_series_data.mean(axis=1)
        time_series_data = time_series_data[['id','time','media_sensor']]
        return time_series_data
    
    def add_tte(self, features, tte_file):
        features_df = pd.read_csv(features)
        tte = pd.read_csv(tte_file)
        tte.columns = ['segment_id', 'time_to_eruption']

        dataframe = features_df.merge(tte, left_on='volcan_id', right_on='segment_id', how='left')
        dataframe.drop(columns=['segment_id'], inplace=True)

        output_dir = "processed"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "tsfresh_data_tte.csv")
        dataframe.to_csv(filepath, index=False)
        print(f"Características con time_to_eruption guardadas en {filepath}")

        return dataframe

if __name__ == '__main__':
    feature_extractor = FeatureExtractor()
    directory = "../data/kaggle/input/train"
    features = "processed/tsfresh_data_minfc.csv"
    tte_file = "../data/kaggle/input/train.csv"
    #extracted_features = feature_extractor.extract_features(directory)
    feature_extractor.add_tte(features, tte_file)

#TODO MLP (Neuronal), ADABOOST y XGBoost mirarmelo
