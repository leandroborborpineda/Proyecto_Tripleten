
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_preprocessing_script import load_data, correlation_matrix_code_2, correlation_matrix_code, clean_data
from graphs.data_graphs import hist_contract_dur, temporal_series_active_contracts, correlation_matrix_graph
from graphs.data_graphs import correlation_matrix_graph_2, service_clients_count, contract_status_graph
from scripts.data_io import save_dataframe
def main():
    
    ## Cargando los datos y agregando información
    data = load_data()

    #Histograma de duración de contrato
    hist_contract_dur(data)
    #Serie temporal para contratos activos
    temporal_series_active_contracts(data)
    #Matriz de correlación 1
    correlation_matrix = correlation_matrix_code(data)
    correlation_matrix_graph(correlation_matrix)

    correlation_matrix_2, total_df_encoded = correlation_matrix_code_2(data)
    correlation_matrix_graph_2(correlation_matrix_2)

    service_clients_count(total_df_encoded)

    data_df_encoded = clean_data(data)

    #print(data_df_encoded)

    contract_status_graph(data_df_encoded)

    data_df_encoded = pd.DataFrame(data_df_encoded)

    save_dataframe(data_df_encoded,'outputs/data/data_df_encoded.csv', format='csv')

    

    
    

if __name__== "__main__":
    main()


