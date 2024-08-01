import pandas as pd
import os

from data_io import load_dataframe

def analyze():
    # Cargar el DataFrame desde el archivo CSV
    total_df = load_dataframe('outputs\data\data_df_encoded.csv', format='csv')
    
    ## Impresion de prueba
    print(total_df)

    

if __name__ == "__main__":
    analyze()