import pandas as pd

# Leer los datos de los archivos CSV
datos_volcanes = pd.read_csv('volcanes_clean.csv')
datos_tiempo_erupcion = pd.read_csv('results.csv')

# Fusionar los datos utilizando el método merge de Pandas
volcanes_con_tiempo_erupcion = pd.concat([datos_volcanes, datos_tiempo_erupcion['time_to_eruption_dt']], axis=1)

# Guardar los datos fusionados en un nuevo archivo CSV
volcanes_con_tiempo_erupcion.to_csv('volcanoes_database.csv', index=False)

print("Done")