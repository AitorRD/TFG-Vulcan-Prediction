import pandas as pd

# Leer el archivo CSV con los datos de los volcanes
datos_volcanes = pd.read_csv('volcano-events.csv')

# Seleccionar solo las columnas necesarias y renombrarlas
datos_volcanes = datos_volcanes[['Name', 'Location', 'Country', 'Latitude', 'Longitude', 'Elevation (m)']]
datos_volcanes.columns = ['Nombre', 'Localización', 'País', 'Latitud', 'Longitud', 'Altura']

# Eliminar filas duplicadas basadas en el nombre del volcán
datos_volcanes = datos_volcanes.drop_duplicates(subset=['Nombre'])

# Guardar los datos limpios en un nuevo archivo CSV
datos_volcanes.to_csv('volcanes_clean.csv', index=False)

print("Done")