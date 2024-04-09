# %%
import pandas as pd
import os

# Directorio que contiene los CSVs
directory = '../../kaggle/input/train'

# Leer train.csv y crear el diccionario segment_id -> time_to_eruption
file_path = '../../kaggle/input/train.csv'
train_df = pd.read_csv(file_path)
train_df['segment_id'] = train_df['segment_id'].astype(str)
time_to_eruption_dict = dict(zip(train_df['segment_id'], train_df['time_to_eruption']))

dfs = []

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(directory, filename))
        media_global = df.mean().mean()
        volcan_id = os.path.splitext(filename)[0]
        volcan_data = {'volcan_id': volcan_id, 'media_global': media_global}
        dfs.append(volcan_data)
        
# Crear DataFrame a partir de la lista de diccionarios
df_global = pd.DataFrame(dfs)
df_global.set_index('volcan_id', inplace=True)

# Asignar time_to_eruption a cada volcan_id en df_global
df_global['time_to_eruption'] = [time_to_eruption_dict.get(segment_id) for segment_id in df_global.index]

# Mostrar el nuevo DataFrame
print(df_global)

# %%
df_global.to_csv('dataframe2.csv', index=True)

# %%



