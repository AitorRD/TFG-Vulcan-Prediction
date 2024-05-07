import pandas as pd
import numpy as np
import os

def calculate_stats(data):
    stats = {}
    stats['max_global'] = data.max().max()
    stats['min_global'] = data.min().min()
    stats['zero_crossings'] = np.count_nonzero(np.diff(np.sign(data), axis=1), axis=1).sum()
    window_size = 600
    num_windows = len(data) // window_size
    for i in range(num_windows):
        window_data = data.iloc[i * window_size : (i + 1) * window_size]
        stats[f'media_global_{i+1}'] = window_data.mean().mean()
        stats[f'desv_tipica_{i+1}'] = window_data.std().mean()
    if len(data) % window_size != 0:
        remaining_data = data.iloc[num_windows * window_size:]
        stats[f'media_global_{num_windows+1}'] = remaining_data.mean().mean()
        stats[f'desv_tipica_{num_windows+1}'] = remaining_data.std().mean()
    
    # Filtrar solo las estadísticas que no estén vacías
    non_empty_stats = {key: value for key, value in stats.items() if not pd.isna(value)}
    
    return non_empty_stats

def process_data(directory, train_file, output_directory):
    train_df = pd.read_csv(train_file)
    train_df['segment_id'] = train_df['segment_id'].astype(str)
    time_to_eruption_dict = dict(zip(train_df['segment_id'], train_df['time_to_eruption']))

    dfs = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            volcan_id = os.path.splitext(filename)[0]
            stats = calculate_stats(df)
            volcan_data = {'volcan_id': volcan_id, **stats}
            volcan_data['time_to_eruption'] = time_to_eruption_dict.get(volcan_id)
            dfs.append(volcan_data)

    df_global = pd.DataFrame(dfs)
    
    # Eliminar columnas vacías
    df_global.dropna(axis=1, how='all', inplace=True)
    
    # Reordenar columnas según el orden especificado
    cols = ['volcan_id', 'time_to_eruption', 'max_global', 'min_global', 'zero_crossings'] + \
           [col for col in df_global.columns if col not in ['volcan_id', 'time_to_eruption', 'max_global', 'min_global', 'zero_crossings']]
    df_global = df_global[cols]

    df_global.set_index('volcan_id', inplace=True)

    output_file = 'dataframe.csv'
    output_path = os.path.join(output_directory, output_file)

    df_global.to_csv(output_path, index=True)
    print('Data processed')

directory = 'src/data/prueba'
train_file = 'src/data/kaggle/input/train.csv'
output_directory = 'src/data/processed'
process_data(directory, train_file, output_directory)