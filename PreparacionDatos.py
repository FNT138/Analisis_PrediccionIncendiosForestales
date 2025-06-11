import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Cargar Excel (ajustar ruta)
ruta_excel = 'datasets/Estadísticas normales Datos abiertos 1991-2020- TODAS HOJAS.xlsx'
clima = pd.read_excel(ruta_excel, header=None)

print("Primeras filas del dataset original:")
print(clima.head(10))

# Extraer nombres de meses
meses = list(clima.iloc[0, 1:].values)

# Variables climáticas (columna 0 desde fila 1)
variables = list(clima.iloc[1:, 0].values)

# Provincia (rellenar NaNs hacia adelante)
provincias_col = clima.iloc[:, 0].ffill()


# Datos numéricos (todas las columnas excepto la primera)
datos = clima.iloc[:, 1:]

# Construcción formato largo
filas = []
for i in range(datos.shape[0]):
    provincia = provincias_col.iloc[i]
    for j in range(datos.shape[1]):
        mes = meses[j]
        variable = variables[i]
        valor = datos.iat[i, j]
        filas.append([provincia.strip() if isinstance(provincia, str) else provincia,
                      mes,
                      variable.strip() if isinstance(variable, str) else variable,
                      valor])

df_largo = pd.DataFrame(filas, columns=['Provincia', 'Mes', 'Variable', 'Valor'])

# Pivotar para formato ancho
clima_wide = df_largo.pivot_table(index=['Provincia', 'Mes'], columns='Variable', values='Valor').reset_index()

# Convertir a numérico las columnas de variables
for col in clima_wide.columns[2:]:
    clima_wide[col] = pd.to_numeric(clima_wide[col], errors='coerce')

# Variables para agrupar y cómo agregarlas
variables_agregar = {
    'Temperatura (°C)': 'mean',
    'Temperatura máxima (°C)': 'mean',
    'Temperatura mínima (°C)': 'mean',
    'Precipitación (mm)': 'sum',
    'Humedad relativa (%)': 'mean',
    'Nubosidad total (octavos)': 'mean',
    'Frecuencia de días con Precipitación superior a 1.0 mm': 'mean',
    'Velocidad del Viento (km/h) (2011-2020)': 'mean'
}

# Filtrar solo variables que existan en el dataset (para evitar KeyError)
variables_existentes = {k: v for k, v in variables_agregar.items() if k in clima_wide.columns}

clima_anual = clima_wide.groupby('Provincia').agg(variables_existentes).reset_index()

print(f"Datos agrupados por provincia, total provincias: {clima_anual.shape[0]}")

# Exportar CSV con codificación UTF-8
clima_anual.to_csv('clima_anual.csv', index=False, encoding='utf-8')

print("Archivo 'clima_anual.csv' generado correctamente.")






# Cargar y procesar otros datasets
def load_and_process(file_path, sheet_name, value_vars=None):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.replace(['s/d', 'NA'], np.nan)
    
    if value_vars:
        df = df.melt(id_vars=['jurisdicción'], 
                     var_name='año', 
                     value_name=value_vars)
        df['año'] = df['año'].str.extract('(\d+)').astype(int)
    
    return df

# Cargar todos los datasets
incendios_sup = load_and_process('Superficie_afectada.xlsx', 'rii_c_hectareas_incendios_prov_', 'hectareas_quemadas')
focos_calor = load_and_process('Cantidad_focos_calor.xlsx', 'Cantidad de focos de calor', 'focos_calor')
densidad = load_and_process('Densidad_poblacion.xlsx', 'resumen')[['provincia', 'prom_densidad_hab_km2']]
incendios_mes = load_and_process('Cantidad_incendios_mes.xlsx', 'rii_a_cantidad_incendios_mes_20')

# Unificar todos los datos
df = pd.merge(incendios_sup, focos_calor, on=['jurisdicción', 'año'])
df = pd.merge(df, densidad, left_on='jurisdicción', right_on='provincia', how='left')
df = pd.merge(df, clima_anual, left_on='jurisdicción', right_on='Provincia', how='left')

# Calcular variables derivadas
df['intensidad_incendio'] = df['hectareas_quemadas'] / df['focos_calor'].replace(0, 1)
df['meses_criticos'] = df['año'].apply(lambda x: 1 if x in [2020, 2022, 2024] else 0)  # Años con mayor actividad

# Crear variables de lag
df['focos_lag1'] = df.groupby('jurisdicción')['focos_calor'].shift(1)
df['hectareas_lag1'] = df.groupby('jurisdicción')['hectareas_quemadas'].shift(1)

# Eliminar valores faltantes
df = df.dropna(subset=['focos_lag1', 'hectareas_lag1'])