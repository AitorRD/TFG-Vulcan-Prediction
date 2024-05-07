from sklearn.tree import DecisionTreeRegressor

def train_dt(X_train_list, y_train_list, random_state, max_depth, min_samples_split, min_samples_leaf):
    tree_model = DecisionTreeRegressor(random_state=random_state, 
                                       max_depth=max_depth, 
                                       min_samples_split=min_samples_split, 
                                       min_samples_leaf=min_samples_leaf)
    for X_train, y_train in zip(X_train_list, y_train_list):
        tree_model.fit(X_train, y_train)
    print('Decision Tree Trained')
    return tree_model




