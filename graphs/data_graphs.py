import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

from data.data_preprocessing_script import correlation_matrix_code

## Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd



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

def contract_status_graph(data_df_encoded):

    estado_contrato = data_df_encoded['contract_status'].value_counts()

    # Crear el gráfico de barras
    plt.figure(figsize=(8, 6))
    estado_contrato.plot(kind='bar', color='skyblue')
    plt.title('Distribución de estados de contrato')
    plt.xlabel('Estado de contrato')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig('outputs/figures/contract_status_graph.png')
    plt.close()



def features_importances_graph(features_train, target_train):

    model_rf = RandomForestClassifier(max_depth=10, n_estimators = 150, random_state=0)
    model_rf.fit(features_train, target_train)

    # Obtener las importancias de las características
    importances_rf = model_rf.feature_importances_

    # Crear un DataFrame para las importancias
    feature_importances_df_rf = pd.DataFrame({
        'Feature': features_train.columns,
        'Importance': importances_rf
    }).sort_values(by='Importance', ascending=False)

    # Visualizar la importancia de las características
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df_rf['Feature'], feature_importances_df_rf['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances (Random Forest)')
    plt.gca().invert_yaxis()
    
    plt.savefig('outputs/figures/cofeatures_importances_graph.png')
    plt.close()