import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Entrenamiento del Mejor Modelo con Todos los Datos de Entrenamiento
data_file = 'src/data/processed/dataframe.csv'
df_train = pd.read_csv(data_file)
X_train = df_train.drop(columns=['time_to_eruption', 'volcan_id'])
y_train = df_train['time_to_eruption']

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_train)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

best_model = GradientBoostingRegressor(max_depth=23,
                                n_estimators=70, 
                                learning_rate=1.0, 
                                loss='squared_error', 
                                random_state= 42)
best_model.fit(X_scaled, y_train)

test_file = 'src/data/processed/dataframe_test.csv'
df_test = pd.read_csv(test_file)
X_test = df_test.drop(columns=['volcan_id'])
segment_ids = df_test['volcan_id']

X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

predictions = best_model.predict(X_test_scaled)

# Preparar el Archivo de Submission
submission_df = pd.DataFrame({
    'segment_id': segment_ids,
    'time_to_eruption': predictions
})

submission_file = 'src/predict/results/submission.csv'
submission_df.to_csv(submission_file, index=False)
print("Submission file created.")