# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:54:44 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Título de la aplicación
st.title('Pronóstico de Holt-Winters Multiplicativo')

# Cargar datos
st.sidebar.header('Cargar datos')
uploaded_file = st.sidebar.file_uploader("Elija un archivo CSV", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Fecha'] = pd.to_datetime(data['Fecha'])  # Asegúrate de que la columna de fecha esté en el formato correcto
    data.set_index('Fecha', inplace=True)

    # Filtrar los datos para mostrar solo desde 2021
    data = data[data.index >= '2021-01-01']

    st.subheader('Datos cargados')
    st.write(data)

    # Mostrar estadísticas descriptivas
    st.subheader('Estadísticas Descriptivas')
    st.write(data.describe())

    # Seleccionar la columna para el pronóstico
    column = st.sidebar.selectbox("Selecciona la columna para el pronóstico", data.columns)

    # Configuración de parámetros
    st.sidebar.header('Parámetros de Holt-Winters')
    seasonal_periods = st.sidebar.number_input("Períodos estacionales (por ejemplo, 12 para datos mensuales)", min_value=1, value=12)
    alpha = st.sidebar.slider("Alpha (nivel)", 0.0, 1.0, 0.2)
    beta = st.sidebar.slider("Beta (tendencia)", 0.0, 1.0, 0.2)
    gamma = st.sidebar.slider("Gamma (estacionalidad)", 0.0, 1.0, 0.2)

    # Ajustar el modelo Holt-Winters
    model = ExponentialSmoothing(data[column], 
                                  seasonal='multiplicative', 
                                  seasonal_periods=seasonal_periods).fit(smoothing_level=alpha, 
                                                                          smoothing_trend=beta, 
                                                                          smoothing_seasonal=gamma)

    # Pronóstico
    forecast_periods = 12  # Pronóstico para los próximos 12 meses
    forecast = model.forecast(forecast_periods)

    # Crear un índice de fechas para el pronóstico (próximos 12 meses)
    forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_periods, freq='M')
    forecast.index = forecast_index

    # Calcular la predicción ajustada en los datos originales
    data['Ajuste'] = model.fittedvalues  # Ajuste de los valores originales

    # Calcular media móvil para la tendencia
    data['Tendencia (Media Móvil)'] = data[column].rolling(window=seasonal_periods).mean()

    # Calcular la estacionalidad usando seasonal_decompose
    decomposition = seasonal_decompose(data[column], model='multiplicative', period=seasonal_periods)
    data['Estacionalidad'] = decomposition.seasonal

    # Calcular residuos
    data['Residuos'] = data[column] - data['Ajuste']

    # Mostrar el DataFrame de resultados
    st.subheader('Resultados del Análisis')
    results_df = data[[column, 'Ajuste', 'Tendencia (Media Móvil)', 'Estacionalidad', 'Residuos']]
    st.write(results_df)

    # Mostrar el DataFrame del pronóstico
    st.subheader('Pronóstico de Holt-Winters')
    forecast_df = pd.DataFrame({'Pronóstico': forecast})
    st.write(forecast_df)

    # Graficar los resultados
    st.subheader('Gráfico del Pronóstico')
    plt.figure(figsize=(10, 5))
    plt.plot(data[column], label='Datos Históricos', color='blue')
    plt.plot(data['Ajuste'], label='Ajuste (Holt-Winters)', linestyle='--', color='orange')
    plt.plot(data['Tendencia (Media Móvil)'], label='Tendencia (Media Móvil)', linestyle='--', color='green')
    plt.plot(forecast, label='Pronóstico', linestyle='--', color='red')

    # Limitar el eje x para mostrar todo desde 2021 hasta el final del pronóstico
    start_date = pd.Timestamp('2021-01-01')
    end_date = forecast.index[-1]
    plt.xlim(start_date, end_date)

    plt.legend()
    st.pyplot(plt)

    # Gráfico: Valores Reales vs Ajuste
    st.subheader('Valores Reales vs Ajuste (Holt-Winters)')
    plt.figure(figsize=(10, 5))
    plt.plot(data[column], label='Datos Reales', color='blue')
    plt.plot(data['Ajuste'], label='Ajuste (Holt-Winters)', linestyle='--', color='orange')

    plt.xlim(start_date, end_date)
    plt.legend()
    st.pyplot(plt)

    # Gráfico: Valores Reales vs Pronóstico
    st.subheader('Valores Reales vs Pronóstico')
    plt.figure(figsize=(10, 5))
    plt.plot(data[column], label='Datos Reales', color='blue')
    plt.plot(forecast, label='Pronóstico', linestyle='--', color='red')

    plt.xlim(start_date, end_date)
    plt.legend()
    st.pyplot(plt)

else:
    st.warning('Por favor, cargue un archivo CSV para continuar.')

# Sección de Ayuda
st.sidebar.header('Ayuda')
st.sidebar.write("""
Esta aplicación realiza pronósticos utilizando el modelo Holt-Winters multiplicativo. A continuación se describen las funcionalidades disponibles:

1. **Cargar datos**: Suba un archivo CSV con datos que contengan una columna de fecha y al menos una columna numérica para el pronóstico. Asegúrese de que la columna de fecha esté en el formato adecuado.

2. **Estadísticas descriptivas**: Se muestran estadísticas básicas de los datos cargados, como la media, mediana y desviación estándar.

3. **Selección de columna para pronóstico**: Elija la columna que desea pronosticar desde el menú desplegable.

4. **Configuración de parámetros**: Ajuste los parámetros del modelo Holt-Winters:
   - **Períodos estacionales**: Especifique el número de períodos que representan un ciclo completo (por ejemplo, 12 para datos mensuales).
   - **Alpha (nivel)**: Controla el nivel de suavizamiento.
   - **Beta (tendencia)**: Controla el suavizamiento de la tendencia.
   - **Gamma (estacionalidad)**: Controla el suavizamiento de la estacionalidad.

5. **Pronóstico**: El pronóstico se realiza para los próximos 12 meses.

6. **Visualizaciones**: Se generan gráficos para comparar los datos reales, los ajustes, la tendencia y el pronóstico.

Para más información sobre cómo usar esta aplicación, consulte la documentación o el código fuente.
""")
