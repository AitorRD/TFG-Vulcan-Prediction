import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.model_selection import KFold
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

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

    selected_indices = np.mean(selected_features_indices, axis=0)
    return selected_indices

def save_selected_features(features_df, selected_indices, output_file):
    selected_features_names = features_df.columns[selected_indices.astype(int)].tolist()  # Convert indices to integers
    selected_df = features_df[['volcan_id', 'time_to_eruption'] + selected_features_names]
    selected_df.to_csv(output_file, index=False)
    print(f"Selected features saved to {output_file}")
    print("Selected features:", selected_features_names)  # Print selected features

if __name__ == '__main__':
    data_file = "src/tsfresh/processed/tsfresh_data_tte.csv"
    features_df = load_data(data_file)
    
    X = features_df.drop(columns=['volcan_id', 'time_to_eruption'])
    y = features_df['time_to_eruption']
    
    method = 'f_regression'  # Choose 'f_regression' or 'mutual_info'
    percentile = 42  
    selected_indices = select_features(X, y, method=method, percentile=percentile, n_splits=5)
    
    output_file = "src/tsfresh/processed/tsfresh_dataframe.csv"
    save_selected_features(features_df, selected_indices, output_file)