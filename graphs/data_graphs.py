import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from data.data_preprocessing_script import correlation_matrix_code


def hist_contract_dur(total_df):
    plt.figure(figsize=(10, 6))
    plt.hist(total_df['ContractDuration'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Histograma de Duración del Contrato')
    plt.xlabel('Duración del Contrato (días)')
    plt.ylabel('Número de Clientes')
    plt.grid(True)
    plt.show()

def temporal_series_active_contracts(total_df):
    serie_temporal = total_df.groupby('BeginDate')['estado_contrato'].sum().cumsum()
    plt.figure(figsize=(10, 6))
    serie_temporal.plot()
    plt.title('Evolución de la cantidad de contratos activos')
    plt.xlabel('Fecha de Inicio del Contrato')
    plt.ylabel('Cantidad de Contratos Activos')
    plt.grid(True)
    plt.show()

def correlation_matrix_code_grap():

    correlation_matrix = correlation_matrix_code(total_df=)

      # Visualizar la matriz de correlación utilizando un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Matriz de Correlación entre Variables (OHE)')
    plt.show()    


