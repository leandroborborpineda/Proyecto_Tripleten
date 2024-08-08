import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import randint, uniform
import random

from data_io import load_dataframe
from data.data_split_train_test import data_split
from prediction_models.prediction_models import dummy_model_script

from prediction_models.prediction_models import LogisticRegMod, grid_search_decision_tree, grid_search_random_forest, random_search_random_forest
from prediction_models.prediction_models import random_search_lgbm, grid_search_lgbm, grid_search_catboost, random_search_catboost
from prediction_models.prediction_models import random_search_xgboost, grid_search_xgboost
from graphs.data_graphs import features_importances_graph



def analyze():
    # Cargar el DataFrame desde el archivo CSV
    data_df_encoded = load_dataframe('outputs\data\data_df_encoded.csv', format='csv')
    
    ## Impresion de prueba
    #print(data_df_encoded)

    ## Aplicando la función de data_split a los datos
    [features_train, features_valid, target_train, target_valid] = data_split(data_df_encoded, 0.25)

    ## Llamando al modelo Dummy:
    #dummy_model_script(features_train, features_valid, target_train, target_valid)

    ## LLamando al modelo Regresión Lineal:
    #LogisticRegMod(features_train, features_valid, target_train, target_valid)
    

    ## Modelo: Decision Tree - Grid Search:
    # Parámetros

    # param_grid_dt = {
    #     'max_depth': [None, 5, 50, 100],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }

    # grid_search_decision_tree(features_train, features_valid, target_train, target_valid, param_grid_dt)


    #Model: Random Forest - Grid Search
    # Parámetros

    # param_grid_rf = {
    # 'n_estimators': [100, 150, 200],
    # 'max_depth': [None, 5, 10, 20, 25],
    # 'min_samples_split': [2, 5],
    # 'min_samples_leaf': [1, 2],
    # }

    # grid_search_random_forest(features_train, features_valid, target_train, target_valid, param_grid_rf)


    #Model: Random Forest - Random Search
    # Parámetros

    # param_dist_rf = {
    # 'n_estimators': randint(50, 150, 200),
    # 'max_depth': [None, 5, 10, 15],
    # 'min_samples_split': randint(2, 11),
    # 'min_samples_leaf': randint(1, 5)
    # }

    # random_search_random_forest(features_train, features_valid, target_train, target_valid, param_dist_rf)


    # Revisando la importancia de las características:
    #features_importances_graph(features_train, target_train)


    # #Model: Light GBM - Random Search
    # # Parámetros

    # # Ejemplo de parámetros a probar en la búsqueda aleatoria para LightGBM
    # param_dist_lgbm = {
    #     'num_leaves': randint(20, 150),
    #     'max_depth': randint(3, 15),
    #     'learning_rate': uniform(0.01, 0.2),
    #     'n_estimators': randint(50, 200),
    #     'min_child_samples': randint(20, 500)
    #     #'subsample': uniform(0.5, 1.0),
    #     #'colsample_bytree': uniform(0.5, 1.0),
    #     #'reg_alpha': uniform(0, 1.0),
    #     #'reg_lambda': uniform(0, 1.0)
    # }

    # random_search_lgbm(features_train, features_valid, target_train, target_valid, param_dist_lgbm, n_iter=100)

    # #Model: Light GBM - Grid Search
    # # Parámetros:

    # param_grid_lgbm = {
    #     'num_leaves': [31, 50, 70],
    #     'max_depth': [-1, 10, 20],
    #     'learning_rate': [0.1, 0.05, 0.01],
    #     'n_estimators': [100, 150, 200],
    #     'min_child_samples': [20, 50, 100]
    # }

    # grid_search_lgbm(features_train, features_valid, target_train, target_valid, param_grid_lgbm, n_iter=100)


    # #Model: Catboost - Grid Search
    # # Parámetros:

#     # Definir el grid de hiperparámetros a buscar
#     param_grid_cb = {
#         'iterations': [100, 200, 250],
#         'depth': [6, 8, 10],
#         'learning_rate': [0.1, 0.05, 0.01],
#         'l2_leaf_reg': [1, 3, 5, 7],
#         'border_count': [32, 50, 100]
# }
    
#     grid_search_catboost(features_train, features_valid, target_train, target_valid, param_grid_cb)



    # #Model: Catboost - Random Search
    # # Parámetros:

    # print("Modelo Catboost - Random Search")

    # # Ejemplo de parámetros a probar en la búsqueda aleatoria para CatBoost
    # param_dist_cb = {
    #     'iterations': randint(10, 100),
    #     'depth': randint(4, 10),
    #     'learning_rate': uniform(0.01, 0.3),
    #     'l2_leaf_reg': uniform(1, 10),
    #     'border_count': randint(32, 255)
    # }


    # random_search_catboost(features_train, features_valid, target_train, target_valid, param_dist_cb, n_iter=100)



#     # #Model: XGBoost - Random Search
    
#     print("Modelo XGBoost - Random Search")

#     # # Parámetros:

# # Ejemplo de parámetros a probar en la búsqueda aleatoria para XGBoost
#     param_dist_xgb = {
#         'n_estimators': randint(50, 200),
#         'max_depth': randint(3, 10),
#         'learning_rate': uniform(0.01, 0.3)
#         #'subsample': uniform(0.5, 1),
#         #'colsample_bytree': uniform(0.5, 1)
#         #'gamma': uniform(0, 0.5),
#         #'reg_alpha': uniform(0, 1),
#         #'reg_lambda': uniform(0, 1)
#     }

#     random_search_xgboost(features_train, features_valid, target_train, target_valid, param_dist_xgb, n_iter=100)


#     # #Model: XGBoost - Grid Search
    
    print("Modelo XGBoost - Grid Search")

# Definir el grid de hiperparámetros a buscar
    param_grid_xgb = {
        'n_estimators': [100, 150, 200],
        'max_depth': [6, 10, 15],
        'learning_rate': [0.1, 0.05, 0.01],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    grid_search_xgboost(features_train, features_valid, target_train, target_valid, param_grid_xgb)

if __name__ == "__main__":
    analyze()

