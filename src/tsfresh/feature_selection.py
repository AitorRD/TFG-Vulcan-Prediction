import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

# Crear directorio si no existe
output_dir = 'src/graphs/feature_selection'
os.makedirs(output_dir, exist_ok=True)

def divide_data_kfold(data_file, n_splits=5):
    df = pd.read_csv(data_file)
    X = df.drop(columns=['time_to_eruption', 'volcan_id'])
    y = df['time_to_eruption']

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X_imputed[train_index], X_imputed[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return X_train_list, X_test_list, y_train_list, y_test_list, X_imputed, y, X

def feature_selection_kfold(X_train_list, y_train_list, X, method='kbest', param=10):
    feature_names = X.columns
    selected_features = []

    for X_train, y_train in zip(X_train_list, y_train_list):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        if method == 'kbest':
            selector = SelectKBest(score_func=f_regression, k=param)
        elif method == 'percentile':
            selector = SelectPercentile(score_func=f_regression, percentile=param)
        else:
            raise ValueError("Método no soportado. Usa 'kbest' o 'percentile'.")

        selector.fit(X_train_scaled, y_train)
        selected_features.append(feature_names[selector.get_support()])

    selected_features = pd.Series(np.concatenate(selected_features)).value_counts()
    selected_features = selected_features[selected_features == len(X_train_list)].index

    return selected_features

data_file = 'src/tsfresh/processed/tsfresh_data_tte.csv'
X_train_list, X_test_list, y_train_list, y_test_list, X_imputed, y, X = divide_data_kfold(data_file, n_splits=5)
selected_features_kbest = feature_selection_kfold(X_train_list, y_train_list, X, method='kbest', param=5)
selected_features_percentile = feature_selection_kfold(X_train_list, y_train_list,X, method='percentile', param=20)

print("Características seleccionadas por SelectKBest:")
print(selected_features_kbest)

print("\nCaracterísticas seleccionadas por SelectPercentile:")
print(selected_features_percentile)

df = pd.read_csv(data_file)
df_selected_kbest = df[['time_to_eruption', 'volcan_id'] + list(selected_features_kbest)]
df_selected_percentile = df[['time_to_eruption', 'volcan_id'] + list(selected_features_percentile)]
df_selected_kbest.to_csv('src/tsfresh/processed/tsfresh_data_tte_selected_kbest.csv', index=False)
df_selected_percentile.to_csv('src/tsfresh/processed/tsfresh_data_tte_selected_percentile.csv', index=False)