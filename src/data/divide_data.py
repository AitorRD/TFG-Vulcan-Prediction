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

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    print("KNN -> Error absoluto medio (MAE):", mae)
    print("KNN -> Error cuadrático medio (MSE):", mse)
    print("KNN -> Error porcentaje absoluto medio (MAPE):", mape)

divide_data('src/data/processed/dataframe.csv')
