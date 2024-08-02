import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_io import load_dataframe
from data.data_split_train_test import data_split
from prediction_models.prediction_models import dummy_model_script

from prediction_models.prediction_models import LogisticRegMod

def analyze():
    # Cargar el DataFrame desde el archivo CSV
    data_df_encoded = load_dataframe('outputs\data\data_df_encoded.csv', format='csv')
    
    ## Impresion de prueba
    print(data_df_encoded)

    ## Aplicando la función de data_split a los datos
    [features_train, features_valid, target_train, target_valid] = data_split(data_df_encoded, 0.25)

    ## Llamando al modelo Dummy:
    #dummy_model_script(features_train, features_valid, target_train, target_valid)

    ## LLamando al modelo Regresión Lineal:
    LogisticRegMod(features_train, features_valid, target_train, target_valid)


if __name__ == "__main__":
    analyze()

