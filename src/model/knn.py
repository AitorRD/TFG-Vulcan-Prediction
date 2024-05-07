from sklearn.neighbors import KNeighborsRegressor

def train_knn(X_train, y_train, k):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    print('KNN Trained')
    return knn
