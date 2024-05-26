import pandas as pd
import numpy as np

data_file = "src/predict/results/results_dt_kfold_default.csv"
eruption_times = pd.read_csv(data_file)

# Generar coordenadas aleatorias
np.random.seed(42)  # Fijar la semilla para reproducibilidad
n_volcanoes = eruption_times.shape[0]
latitudes = np.random.uniform(-90, 90, n_volcanoes)
longitudes = np.random.uniform(-180, 180, n_volcanoes)

# Añadir las coordenadas al dataset original
eruption_times['latitude'] = latitudes
eruption_times['longitude'] = longitudes

# Guardar el dataset modificado
output_file = 'src/data/mapbox/volcanoes_database.csv'
eruption_times.to_csv(output_file, index=False)
print(f"Dataset saved as '{output_file}'")