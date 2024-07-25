
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_preprocessing_script import data_preprocessing
from data.data_preprocessing_script import prueba
from graphs.data_graphs import hist_contract_dur, temporal_series_active_contracts, correlation_matrix_graph



def main():
    total_df = data_preprocessing()
    
    #print(total_df)

    #hist_contract_dur(total_df)

    #temporal_series_active_contracts(total_df)

    

    correlation_matrix_graph()






if __name__== "__main__":
    main()


