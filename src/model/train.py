from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor #TODO Bayesian optimization

def train_knn(X_train_list, y_train_list, k):
    knn = KNeighborsRegressor(n_neighbors=k)
    for X_train, y_train in zip(X_train_list, y_train_list):
        knn.fit(X_train, y_train)
    print('KNN Trained')
    return knn

def train_dt(X_train_list, y_train_list, random_state, max_depth, min_samples_split, min_samples_leaf):
    tree_model = DecisionTreeRegressor(random_state=random_state, 
                                       max_depth=max_depth, 
                                       min_samples_split=min_samples_split, 
                                       min_samples_leaf=min_samples_leaf)
    for X_train, y_train in zip(X_train_list, y_train_list):
        tree_model.fit(X_train, y_train)
    print('Decision Tree Trained')
    return tree_model


def train_rf(X_train_list, y_train_list, random_state, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, 
                                     random_state=random_state, 
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf)
    for X_train, y_train in zip(X_train_list, y_train_list):
        rf_model.fit(X_train, y_train)
    print('Random Forest Trained')
    return rf_model