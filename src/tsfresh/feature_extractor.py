from tsfresh import extract_features, extract_relevant_features, MinimalFCParameters
from sklearn import feature_selection, feature_extraction #TODO hacer esto tmb 
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, time_series_data):
        extracted_features = extract_features(time_series_data, column_id='volcan_id', default_fc_parameters=MinimalFCParameters) #TODO argumento para limitar las features

        return extracted_features

feature_extractor = FeatureExtractor()
data_file = "../data/processed/dataframe.csv"
df = pd.read_csv(data_file)
extracted_features = feature_extractor.extract_features(df)

#TODO MLP (Neuronal), ADABOOST y XGBoost mirarmelo