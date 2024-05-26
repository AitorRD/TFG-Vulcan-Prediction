import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos
df = pd.read_csv('src/tsfresh/processed/tsfresh_data_tte.csv ')

# Separar características (X) y variable objetivo (y)
X = df.drop(columns=['volcan_id', 'time_to_eruption'])
y = df['time_to_eruption']

# Escalar características para L1-based feature selection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# L1-based feature selection (Lasso)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_scaled, y)
model_lasso = SelectFromModel(lasso, prefit=True)
X_lasso = model_lasso.transform(X)

# Tree-based feature selection (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
model_rf = SelectFromModel(rf, prefit=True)
X_rf = model_rf.transform(X)

# Obtener las características seleccionadas
selected_features_lasso = X.columns[model_lasso.get_support()]
selected_features_rf = X.columns[model_rf.get_support()]

# Obtener las características no seleccionadas
not_selected_features_lasso = X.columns[~model_lasso.get_support()]
not_selected_features_rf = X.columns[~model_rf.get_support()]

print("Características seleccionadas por L1-based (Lasso):")
print("Seleccionadas:", selected_features_lasso)
print("No seleccionadas:", not_selected_features_lasso)

print("\nCaracterísticas seleccionadas por Tree-based (Random Forest):")
print("Seleccionadas:", selected_features_rf)
print("No seleccionadas:", not_selected_features_rf)

# Eliminar las columnas no seleccionadas del conjunto de datos original
df_selected_lasso = df.drop(columns=not_selected_features_lasso)
df_selected_rf = df.drop(columns=not_selected_features_rf)

# Guardar los conjuntos de datos modificados en archivos CSV
df_selected_lasso.to_csv('src/tsfresh/processed/tsfresh_data_tte_selected_lasso.csv', index=False)
df_selected_rf.to_csv('src/tsfresh/processed/tsfresh_data_tte_selected_rf.csv', index=False)

#===================================================================================================================
'''import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, f_regression, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

csv_file_path = 'src/tsfresh/processed/tsfresh_data_tte.csv '
df = pd.read_csv(csv_file_path)
results = {}

X = df.drop(columns=['volcan_id', 'time_to_eruption'])
y = df['time_to_eruption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_feature_selection(selector, name):
    model = Pipeline([
        ('selector', selector),
        ('regressor', LinearRegression())
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mean_mae = -np.mean(scores)
    results[name] = mean_mae
    print(f'{name} - MAE: {-results[name]}')

selector = VarianceThreshold(threshold=(.8 * (1 - .8)))  
evaluate_feature_selection(selector, 'VarianceThreshold')

selector = SelectKBest(score_func=f_regression, k=5)  
evaluate_feature_selection(selector, 'SelectKBest')

selector = SelectPercentile(score_func=f_regression, percentile=10) 
evaluate_feature_selection(selector, 'SelectPercentile')

selector = SelectFromModel(LassoCV(cv=5))
evaluate_feature_selection(selector, 'L1-based')

selector = SelectFromModel(RandomForestRegressor(n_estimators=100))
evaluate_feature_selection(selector, 'Tree-based')

best_method = min(results, key=results.get)
print(f'\nMejor método: {best_method} con un MAE de {-results[best_method]}')'''