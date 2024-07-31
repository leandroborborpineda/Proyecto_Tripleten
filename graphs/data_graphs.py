import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

from data.data_preprocessing_script import correlation_matrix_code


def hist_contract_dur(total_df):

    os.makedirs('outputs/figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(total_df['ContractDuration'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Histograma de Duración del Contrato')
    plt.xlabel('Duración del Contrato (días)')
    plt.ylabel('Número de Clientes')
    plt.grid(True)
    plt.savefig('outputs/figures/hist_contract_dur.png')
    plt.close()

def temporal_series_active_contracts(total_df):

    os.makedirs('outputs/figures', exist_ok=True)
    serie_temporal = total_df.groupby('BeginDate')['estado_contrato'].sum().cumsum()
    plt.figure(figsize=(10, 6))
    serie_temporal.plot()
    plt.title('Evolución de la cantidad de contratos activos')
    plt.xlabel('Fecha de Inicio del Contrato')
    plt.ylabel('Cantidad de Contratos Activos')
    plt.grid(True)
    plt.savefig('outputs/figures/temporal_series_active_contracts.png')
    plt.close()

def correlation_matrix_graph(correlation_matrix):
    os.makedirs('outputs/figures', exist_ok=True)
    # Visualizar la matriz de correlación utilizando un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Matriz de Correlación entre Variables (OHE)')
    plt.savefig('outputs/figures/correation_matrix.png')
    plt.close()

def correlation_matrix_graph_2(correlation_matrix_2):
    os.makedirs('outputs/figures', exist_ok=True)
    # Visualizar la matriz de correlación utilizando un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_2, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Matriz de Correlación entre Variables (OHE)')
    plt.savefig('outputs/figures/correation_matrix_2.png')
    plt.close()

def service_clients_count(total_df_encoded):
    # Calcular la cantidad de clientes por servicio
    services_count = total_df_encoded[['OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']].sum()

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    services_count.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Cantidad de Clientes por Servicio')
    plt.xlabel('Servicio')
    plt.ylabel('Número de Clientes')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('outputs/figures/service_clients_count.png')
    plt.close()

