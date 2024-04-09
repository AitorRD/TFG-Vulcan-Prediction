import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def train_and_predict_knn(data_file):
    # Paso 1: Cargar los datos
    df = pd.read_csv(data_file)

    # Paso 2: Dividir los datos en características (X) y etiquetas (y)
    X = df.drop(columns=['time_to_eruption', 'volcan_id'])
    y = df['time_to_eruption']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Paso 3: Imputar valores faltantes con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Paso 4: Escalar características (features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Paso 5: Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Paso 6: Entrenar modelo KNN
    k = 5
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Paso 7: Predecir en el conjunto de prueba
    y_pred = knn.predict(X_test)

    # Paso 8: Calcular el error absoluto medio (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("KNN -> Error absoluto medio (MAE):", mae)
    print("KNN -> Error cuadrático medio (MSE):", mse)


