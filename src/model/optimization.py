import pandas as pd
from sklearn.impute import SimpleImputer
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

data_file = "src/data/processed/dataframe.csv"
df = pd.read_csv(data_file)
X = df.drop(columns=['time_to_eruption', 'volcan_id'])
y = df['time_to_eruption']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

def params_dt(max_depth, min_samples_split):
    model = DecisionTreeRegressor(
        max_depth=int(max_depth), 
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    score = -cross_val_score(model, X_imputed, y, cv=5, scoring='neg_mean_absolute_error').mean()
    return score

def params_rf(n_estimators, max_depth, min_samples_split):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators), 
        max_depth=int(max_depth), 
        min_samples_split=int(min_samples_split), 
        random_state=42,
        n_jobs=-1
    )
    score = -cross_val_score(model, X_imputed, y, cv=5, scoring='neg_mean_absolute_error').mean()
    return score

def params_ab(n_estimators, learning_rate):
    estimator = DecisionTreeRegressor(max_depth=5)
    model = AdaBoostRegressor(
        estimator=estimator,
        n_estimators=int(n_estimators), 
        learning_rate=learning_rate, 
        loss='linear',
        random_state=42
    )
    score = -cross_val_score(model, X_imputed, y, cv=5, scoring='neg_mean_absolute_error').mean()
    return score

def optimize_dt():
    print("______________________DECISION TREE OPTIMIZING______________________")
    dt_param_bounds = {
        'max_depth': (1, 50),
        'min_samples_split': (2, 20)
    }

    dt_optimizer = BayesianOptimization(
        f=params_dt,
        pbounds=dt_param_bounds,
        random_state=42,
        allow_duplicate_points=False
    )
    dt_optimizer.maximize(init_points=10, n_iter=40)
    print("Best hyperparameters for DecisionTreeRegressor: ", dt_optimizer.max)

def optimize_rf():
    print("______________________RANDOM FOREST OPTIMIZING______________________")
    rf_param_bounds = {
        'n_estimators': (2, 1000),
        'max_depth': (1, 50),
        'min_samples_split': (2, 20)
    }

    rf_optimizer = BayesianOptimization(
        f=params_rf,
        pbounds=rf_param_bounds,
        random_state=42,
        allow_duplicate_points=False
    )
    rf_optimizer.maximize(init_points=10, n_iter=40)
    print("Best hyperparameters for RandomForestRegressor: ", rf_optimizer.max)

def optimize_ab():
    print("______________________ADABOOST OPTIMIZING______________________")    
    ab_param_bounds = {
        'n_estimators': (2, 500),
        'learning_rate': (0.01, 1.0)
    }

    ab_optimizer = BayesianOptimization(
        f=params_ab,
        pbounds=ab_param_bounds,
        random_state=42,
        allow_duplicate_points=False
    )
    ab_optimizer.maximize(init_points=10, n_iter=40)
    print("Best hyperparameters for AdaBoostRegressor: ", ab_optimizer.max)