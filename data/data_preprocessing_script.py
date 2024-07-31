import pandas as pd
from functools import reduce
import os
import sys
import math
from scipy import stats as st
import time
import random 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def load_data():
    contract_df = pd.read_csv('datasets/final_provider/contract.csv')
    internet_df = pd.read_csv('datasets/final_provider/internet.csv')
    personal_df = pd.read_csv('datasets/final_provider/personal.csv')
    phone_df = pd.read_csv('datasets/final_provider/phone.csv')

    # Crear una lista de DataFrames
    dataframes = [contract_df, internet_df, personal_df, phone_df]

    # Concatenar los DataFrames a lo largo del eje de las columnas
    total_df = pd.concat(dataframes, axis=1)

    # Asegurarse de que el índice sea el 'customerID'
    total_df = total_df.loc[:,~total_df.columns.duplicated()]

    #print(total_df)

    print("Imprimiendo informcación")
    print(total_df.info())

    ## Cambiando el tipo de dato en las fechas:
    total_df['BeginDate'] = pd.to_datetime(total_df['BeginDate'])
    total_df['EndDate'] = pd.to_datetime(total_df['EndDate'], format='%Y-%m-%d', errors = 'coerce') #  # Coerción para manejar valores no válidos
    total_df['EndDate'].fillna(pd.NaT, inplace = True)

    # Obtener el rango de fechas en la columna BeginDate
    fecha_inicio_min = total_df['BeginDate'].min()
    fecha_inicio_max = total_df['BeginDate'].max()

    # Obtener el rango de fechas en la columna EndDate
    fecha_fin_min = total_df['EndDate'].min()
    fecha_fin_max = total_df['EndDate'].max()

    # # Imprimir los rangos de fechas
    # print("Rango de fechas en la columna BeginDate:")
    # print("Fecha mínima:", fecha_inicio_min)
    # print("Fecha máxima:", fecha_inicio_max)
    # print("\nRango de fechas en la columna EndDate:")
    # print("Fecha mínima:", fecha_fin_min)
    # print("Fecha máxima:", fecha_fin_max)

    ## Cambiando espacios vacíos con NaN
    total_df['TotalCharges'].replace(" ", np.nan, inplace = True)
    total_df['TotalCharges'] = total_df['TotalCharges'].astype(float)

    ## Separando las columnas de servicios
    internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    ## Rellenando los datos ausentes.
    total_df[internet_services] = total_df[internet_services].fillna("No")

    ## Rellenando los datos ausentes
    total_df['estado_contrato'] = total_df['EndDate'].isnull()

    ## Creando la columna de "estado_contrato" donde "EndDate" es un dato ausente
    total_df['estado_contrato'] = total_df['EndDate'].isnull()

    ## Esta columna se añadió para ver los datos desde otro enfoque
    total_df['Cancelado'] = total_df['EndDate'].notnull()

    #### Esta es la fecha final de obtención de los datos: (1 de febrero de 2020)
    last_day = pd.to_datetime('01/02/2020', format='%d/%m/%Y')

    ## COmpletando los datos NA con el ultimo día
    total_df['EndDate_2'] = total_df['EndDate'].fillna(last_day)

    # Calcular la duración del contrato restando la fecha de inicio del contrato de la fecha actual
    total_df['ContractDuration'] = (total_df['EndDate_2'] - total_df['BeginDate']).dt.days

    # Reemplazar los valores NaN en la columna 'ContractDuration' con un valor predeterminado, como -1
    total_df['ContractDuration'].fillna(-1, inplace=True)   

    # Crear la nueva columna 'num_services' que cuenta el número de 'yes' en las columnas de servicios
    total_df['num_services'] = total_df[internet_services].apply(lambda row: row.eq('Yes').sum(), axis=1)

    

    return total_df


def correlation_matrix_code(total_df):
    
    # Aplicar One-Hot Encoding a las variables categóricas
    total_df_encoded = pd.get_dummies(total_df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']], drop_first=True)

    # Calcular la matriz de correlación
    correlation_matrix = total_df_encoded.corr()

    return correlation_matrix


def correlation_matrix_code_2(total_df):
    # Aplicar One-Hot Encoding a las variables categóricas
    total_df_encoded = pd.get_dummies(total_df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                                'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaymentMethod',
                                                'Type', 'PaperlessBilling', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'MultipleLines',
                                                'estado_contrato','TotalCharges', 'MonthlyCharges','BeginDate','EndDate']], drop_first=True)

    # Calcular la matriz de correlación
    correlation_matrix_2 = total_df_encoded.corr()

    return correlation_matrix_2, total_df_encoded


