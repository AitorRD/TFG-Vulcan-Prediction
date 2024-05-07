import pandas as pd
from sklearn.model_selection import train_test_split , KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def divide_data(data_file):
    df = pd.read_csv(data_file)
    X = df.drop(columns=['time_to_eruption', 'volcan_id'])
    y = df['time_to_eruption']

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    print('Data divided')
    return X_train, X_test, y_train, y_test, X_imputed
