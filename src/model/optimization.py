import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data_file = "src/data/processed/dataframe.csv"

def divide_data_kfold(data_file, n_splits=5):
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

    return X_train_list, X_test_list, y_train_list, y_test_list

X_train_list, X_test_list, y_train_list, y_test_list = divide_data_kfold(data_file)

def params_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators), 
        max_depth=int(max_depth), 
        min_samples_split=int(min_samples_split), 
        min_samples_leaf=int(min_samples_leaf), 
        random_state=42
    )
    scores = []
    for X_train, y_train in zip(X_train_list, y_train_list):
        score = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
        scores.append(score)
    return np.mean(scores)

def params_dt(max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeRegressor(
        max_depth=int(max_depth), 
        min_samples_split=int(min_samples_split), 
        min_samples_leaf=int(min_samples_leaf), 
        random_state=42
    )
    scores = []
    for X_train, y_train in zip(X_train_list, y_train_list):
        score = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
        scores.append(score)
    return np.mean(scores)

rf_param_bounds = {
    'n_estimators': (10, 300),
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}

dt_param_bounds = {
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}

# Optimización para RandomForestRegressor
rf_optimizer = BayesianOptimization(
    f=params_rf,
    pbounds=rf_param_bounds,
    random_state=42,
    allow_duplicate_points=True  # Permitir puntos duplicados
)
rf_optimizer.maximize(init_points=10, n_iter=5)

# Optimización para DecisionTreeRegressor
'''dt_optimizer = BayesianOptimization(
    f=params_dt,
    pbounds=dt_param_bounds,
    random_state=42,
    allow_duplicate_points=True  # Permitir puntos duplicados
)
dt_optimizer.maximize(init_points=10, n_iter=10)'''

# Imprimir mejores hiperparámetros
print("Best hyperparameters for RandomForestRegressor: ", rf_optimizer.max)
#print("Best hyperparameters for DecisionTreeRegressor: ", dt_optimizer.max)