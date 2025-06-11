# Analisis y Prediccion de Incendios Forestales en Argentina


## Objetivo General
Desarrollar un modelo predictivo y visual que permita identificar zonas y periodos de alto riesgo de incendios forestales en Argentina, combinando datos climáticos históricos y actuales, junto con información geográfica y social, utilizando **Python** y **Power BI**.

## Objetivos Específicos
* Analizar espacial y temporalmente los incendios forestales ocurridos en Argentina durante los últimos años.

* Integrar variables climáticas históricas y recientes para estudiar su correlación con los focos de incendio.
* Desarrollar modelos de predicción que permitan anticipar zonas de riesgo.
* Evaluar la influencia de variables sociales/geográficas (cercanía a rutas, zonas urbanas, uso del suelo) en la ocurrencia de incendios.
* Visualizar los resultados mediante el dashboard interactivo en **Power BI**.

## Preguntas Claves para la Investigación.
* ¿Dónde y cuándo ocurren con mayor frecuencia los incendios forestales en Argentina?.
* ¿Cuáles son las condiciones climáticas más asociadas a estos eventos?.
* ¿Existen patrones espaciales o temporales que permitan anticipar estos eventos?.
* ¿Qué variables geográficas o sociales aumentan la probabilidad de ocurrencia?.
* ¿Se puede predecir el riesgo de incendios con suficiente precisión a partir de los datos disponibles?.
* ¿Existe relación entre los incendios y la variación de presupuesto en los entes encargados del manejo de los incendios forestales?

### Herramientas
**Python**: procesamiento de datos, modelado estadístico y predicción utilizando Pandas, NumPy, Scikit-learn, XGBoost, Rasterio, GeoPandas, Matplotlib, Seaborn.

**Power BI**: Visualización interactiva


## Resultados esperados
* Análisis estadístico y espacial detallado sobre los incendios en Argentina.

* Modelos predictivos entrenados y validados (clasificación y/o regresión).

* Visualizaciones interactivas con mapas de calor, series temporales y dashboards.

## Modelo Predictivo Propuesto
'''mermaid
flowchart LR
A[Datos históricos] --> B[Preprocesamiento]
B --> C[Entrenamiento]
C --> D((Modelo 1: Random Forest<br>Riesgo categórico))
C --> E((Modelo 2: XGBoost<br>Superficie afectada))
C --> F((Modelo 3: Prophet<br>Tendencia mensual))
D --> G[Evaluación]
E --> G
F --> G
G --> H[Dashboard interactivo]
'''
