import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.impute import SimpleImputer

def divide_data(data_file, n_splits=5):
    df = pd.read_csv(data_file)
    X = df.drop(columns=['time_to_eruption', 'volcan_id'])
    y = df['time_to_eruption']

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    kf = KFold(n_splits=n_splits)

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

    return X_train_list, X_test_list, y_train_list, y_test_list, X_imputed


def divide_data_time_series(data_file, n_splits=5):
    df = pd.read_csv(data_file)
    X = df.drop(columns=['time_to_eruption', 'volcan_id'])
    y = df['time_to_eruption']

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X_imputed[train_index], X_imputed[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return X_train_list, X_test_list, y_train_list, y_test_list, X_imputed