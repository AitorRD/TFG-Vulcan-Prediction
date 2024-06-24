import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def remove_constant_features(X):
    # Inicializar el selector de varianza con un umbral de 0 para eliminar solo características constantes
    selector = VarianceThreshold(threshold=0)
    selector.fit(X)
    
    # Obtener los índices de las características que no son constantes
    non_constant_features = selector.get_support(indices=True)
    
    # Identificar y mostrar las características constantes
    constant_features = [column for column in X.columns if column not in X.columns[non_constant_features]]
    if constant_features:
        print(f"Características constantes eliminadas: {constant_features}")
    else:
        print("No se encontraron características constantes.")
    
    # Retornar el DataFrame sin las características constantes
    return X.iloc[:, non_constant_features]

def select_features(X, y, method='f_regression', percentile=10, n_splits=5):
    selected_features_indices = []
    kf = KFold(n_splits=n_splits)

    for train_index, _ in kf.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]

        if method == 'f_regression':
            selector = SelectPercentile(score_func=f_regression, percentile=percentile)
        elif method == 'mutual_info':
            selector = SelectPercentile(score_func=mutual_info_regression, percentile=percentile)
        else:
            raise ValueError("Invalid method. Use 'f_regression' or 'mutual_info'")

        selector.fit(X_train, y_train)
        selected_features_indices.append(selector.get_support(indices=True))

    # Convertir la lista de índices seleccionados a una matriz numpy y calcular la frecuencia de cada índice
    selected_indices_matrix = np.array(selected_features_indices)
    selected_indices_counts = np.sum(selected_indices_matrix, axis=0)
    
    # Seleccionar características que fueron seleccionadas al menos en la mitad de los pliegues
    selected_indices = np.where(selected_indices_counts >= (n_splits // 2))[0]
    
    return selected_indices

def save_selected_features(features_df, selected_indices, output_file):
    selected_features_names = features_df.columns[selected_indices].tolist()
    selected_df = features_df[['volcan_id', 'time_to_eruption'] + selected_features_names]
    selected_df.to_csv(output_file, index=False)
    print(f"Selected features saved to {output_file}")
    print("Selected features:", selected_features_names)

if __name__ == '__main__':
    data_file = "src/tsfresh/processed/tsfresh_data_tte.csv"
    features_df = load_data(data_file)
    
    X = features_df.drop(columns=['volcan_id', 'time_to_eruption'])
    y = features_df['time_to_eruption']
    
    # Remover características constantes antes de la selección
    X = remove_constant_features(X)
    
    method = 'mutual_info'
    percentile = 42
    selected_indices = select_features(X, y, method=method, percentile=percentile, n_splits=5)
    
    # Ajustar los índices seleccionados al DataFrame original
    selected_features_indices_in_original_df = X.columns[selected_indices].tolist()
    final_selected_indices = [features_df.columns.get_loc(feature) for feature in selected_features_indices_in_original_df]
    
    output_file = "src/tsfresh/processed/tsfresh_dataframe.csv"
    save_selected_features(features_df, final_selected_indices, output_file)