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
from sklearn.preprocessing import StandardScaler


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
    total_df['EndDate'] = pd.to_datetime(total_df['EndDate'], errors = 'coerce') #  # Coerción para manejar valores no válidos
    


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


def clean_data(total_df):
    #Seleccionando y eliminando las columnas indicadas
    data_df = total_df.drop(['customerID', 'BeginDate', 'EndDate', 'gender', 'EndDate_2', 'Cancelado', 'TotalCharges'], axis=1)
    data_df=data_df.rename(columns={'Type':'plan_type', 'PaperlessBilling':'paperless_billing', 'PaymentMethod':'payment_method',
                                'MonthlyCharges':'monthly_charges', 'TotalCharges':'total_charges', 'InternetService':'internet_service',
                                'OnlineSecurity':'online_security','OnlineBackup':'online_backup','DeviceProtection':'device_protection',
                                'TechSupport':'tech_support','StreamingTV':'streaming_tv', 'StreamingMovies':'streaming_movies',
                                'SeniorCitizen':'senior_citizen', 'Partner':'partner', 'Dependents':'dependents', 'MultipleLines':'multiple_lines',
                                'ContractDuration':'contract_duration', 'estado_contrato':'contract_status' })
    
    # Seleccionar solo las columnas categóricas para codificar
    categorical_columns_sna = ['plan_type', 'paperless_billing', 'payment_method', 'online_security','online_backup', 
                           'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'senior_citizen',
                           'partner', 'dependents']

    categorical_columns_na = ['internet_service', 'multiple_lines' ]

    # Realizar one-hot encoding en las columnas categóricas
    data_df_encoded_1 = pd.get_dummies(data_df, columns=categorical_columns_sna, drop_first = True)

    data_df_encoded = pd.get_dummies(data_df_encoded_1, columns=categorical_columns_na)

    # Seleccionar solo las características numéricas para estandarizar
    numeric_features = ['monthly_charges', 'contract_duration', 'num_services']
    numeric_data = data_df_encoded[numeric_features]

    # Inicializar el StandardScaler
    scaler = StandardScaler()

    # Estandarizar las características numéricas
    scaled_data = scaler.fit_transform(numeric_data)

    # Convertir el resultado a DataFrame y asignar nombres de columnas
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_features)

    # Reemplazar las características numéricas originales con las estandarizadas en el DataFrame original
    data_df_encoded[numeric_features] = scaled_df

    
    return data_df_encoded

