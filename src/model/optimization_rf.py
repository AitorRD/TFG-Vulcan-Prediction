import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from bayes_opt import BayesianOptimization

def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, X_train_list, y_train_list, X_test_list, y_test_list):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)

    mse_scores = []

    for X_train, X_test, y_train, y_test in zip(X_train_list, X_test_list, y_train_list, y_test_list):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))

    return -np.mean(mse_scores)

def optimize_random_forest(X_train_list, y_train_list, X_test_list, y_test_list):
    pbounds = {
        'n_estimators': (100, 1000),
        'max_depth': (10, 100),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    }

    optimizer = BayesianOptimization(
        f=lambda n_estimators, max_depth, min_samples_split, min_samples_leaf: rf_cv(
            n_estimators, max_depth, min_samples_split, min_samples_leaf, 
            X_train_list, y_train_list, X_test_list, y_test_list
        ),
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=10, n_iter=30)

    return optimizer.max

def evaluate_model(X_train_list, X_test_list, y_train_list, y_test_list, best_params):
    mse_scores = []
    mae_scores = []
    mape_scores = []

    rf = RandomForestRegressor(
        n_estimators=int(best_params['params']['n_estimators']),
        max_depth=int(best_params['params']['max_depth']),
        min_samples_split=int(best_params['params']['min_samples_split']),
        min_samples_leaf=int(best_params['params']['min_samples_leaf']),
        random_state=42,
        n_jobs=-1
    )

    for X_train, X_test, y_train, y_test in zip(X_train_list, X_test_list, y_train_list, y_test_list):
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))

    mse_mean = np.mean(mse_scores)
    mae_mean = np.mean(mae_scores)
    mape_mean = np.mean(mape_scores)

    return mse_mean, mae_mean, mape_mean