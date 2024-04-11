
import os
import pandas as pd

directorio = './kaggle/input/train'

# Cálculo de medias
medias_por_csv2 = {}
for archivo in os.listdir(directorio):
    if archivo.endswith('.csv'): 
        ruta_csv = os.path.join(directorio, archivo)  
        id = archivo.split('.')[0]
        df = pd.read_csv(ruta_csv)
        media_sensor2 = df.mean()
        medias_por_csv2[id] = media_sensor2

# Cálculo de desviaciones estándar
desv_por_csv = {}
for archivo in os.listdir(directorio):
    if archivo.endswith('.csv'): 
        ruta_csv = os.path.join(directorio, archivo)  
        id = archivo.split('.')[0]
        df = pd.read_csv(ruta_csv)
        desv_sensor = df.std()
        desv_por_csv[id] = desv_sensor

# Crear DataFrames con los resultados
medias_df2 = pd.DataFrame.from_dict(medias_por_csv2, orient='index')
desv_df = pd.DataFrame.from_dict(desv_por_csv, orient='index')

print(medias_df2)

#BOXPLOT DE MEDIAS Y DESV
import matplotlib.pyplot as plt

def create_boxplot(data, title):
    plt.figure(figsize=(10, 6))
    data.boxplot()
    plt.title(title)
    plt.xlabel('Sensor')
    plt.ylabel('Valor')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

create_boxplot(medias_df2, 'Boxplot de las medias de cada sensor')
plt.savefig('graphs/boxplot_medias.png')
plt.show()

create_boxplot(desv_df, 'Boxplot de las desviaciones estándar de cada sensor')
plt.savefig('graphs/boxplot_desviaciones.png')
plt.show()

#BOXPLOT SEABORN
import seaborn as sns
import matplotlib.pyplot as plt

def create_boxplot_seaborn(data, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, palette="Set3")
    plt.title(title)
    plt.xlabel('Sensor')
    plt.ylabel('Valor')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

create_boxplot_seaborn(medias_df2, 'Boxplot de las medias de cada sensor')
plt.savefig('graphs/boxplot_medias_seaborn.png')
plt.show()

create_boxplot_seaborn(desv_df, 'Boxplot de las desviaciones estándar de cada sensor')
plt.savefig('graphs/boxplot_desviaciones_seaborn.png')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

medias_stacked = medias_df2.stack().reset_index()
desv_stacked = desv_df.stack().reset_index()
medias_stacked.columns = ['ID', 'Sensor', 'Valor']
desv_stacked.columns = ['ID', 'Sensor', 'Valor']

plt.figure(figsize=(10, 6))
sns.barplot(data=medias_stacked, x='Sensor', y='Valor')
sns.lineplot(data=desv_stacked, x='Sensor', y='Valor', dashes=False, markers="o", legend=None)
plt.title('Gráfico de Barras con Líneas de Medias y Desviaciones Estándar por Sensor')
plt.xlabel('Sensores')
plt.ylabel('Valor')
plt.tight_layout()
plt.show()

#CONCATENAR DESV Y MEDIAS
num_sensores = 10
medias_df2.columns = [f"media_sensor_{i+1}" for i in range(num_sensores)]
desv_df.columns = [f"desv_sensor_{i+1}" for i in range(num_sensores)]

result = pd.concat([medias_df2, desv_df], axis=1)
result.index.name = 'ID'
print(result)

#AÑADIMOS EL TTE
train_df = pd.read_csv('kaggle/input/train.csv')

result.index = result.index.astype('int64')
dataframe = result.merge(train_df, left_index=True, right_on='segment_id', how='left')
dataframe['time_to_eruption'] = dataframe['time_to_eruption'].where(~dataframe['time_to_eruption'].isna(), None)

print(dataframe)
dataframe.to_csv('dataframe.csv', index=False)

#GUARDAR EL CSV DATAFRAME DE TRAIN
df = pd.read_csv("dataframe.csv")
df.set_index('segment_id', inplace=True, drop=False)

if 'segment_id' in df.columns:
    cols = df.columns.tolist()
    cols = ['segment_id'] + [col for col in cols if col != 'segment_id']
    df = df[cols]

df.to_csv("dataframe_mod.csv")

#BOXPLOT KNN
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("code/knn_iteration.csv")
tte = df["time_to_eruption_pred_knn"]

plt.figure(figsize=(8, 6))
plt.boxplot(tte)
plt.title('Boxplot de time_to_eruption_pred_knn')
plt.xlabel('1º Iteración')
plt.ylabel('TTE')
plt.grid(True)
plt.show()

#TTE BOXPLOT SEABORN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("code/knn_iteration.csv")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='time_to_eruption_pred_knn')
plt.title('Boxplot de time_to_eruption_pred_knn')
plt.xlabel('1º Iteración')
plt.ylabel('TTE')
plt.grid(True)
plt.show()


