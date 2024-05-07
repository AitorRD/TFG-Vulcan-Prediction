from sklearn.neighbors import KNeighborsRegressor

def train_knn(X_train_list, y_train_list, k):
    knn = KNeighborsRegressor(n_neighbors=k)
    for X_train, y_train in zip(X_train_list, y_train_list):
        knn.fit(X_train, y_train)
    print('KNN Trained')
    return knn
