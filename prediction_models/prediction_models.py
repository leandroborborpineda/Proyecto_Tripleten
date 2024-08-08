## Librerias para entrenamiento:

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#from data.data_split_train_test import data_split
from metrics_scripts.metric_scripts_code import sklearn_metrics
from metrics_scripts.metric_scripts_code import recall_presicion_roc

## Modelo Regresión Logística
from sklearn.linear_model import LogisticRegression
## Modelo Decision Tree
from sklearn.tree import DecisionTreeClassifier
## Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
## Modelo Dummy
from sklearn.dummy import DummyClassifier

## Light GBM
import lightgbm as lgb
from scipy.stats import randint, uniform

##Catboost
import catboost as cb

## XGBoost
import xgboost as xgb

## Para agregar una barra de progreso
from tqdm import tqdm






def dummy_model_script(features_train, features_valid, target_train, target_valid):
    
    dummy_model = DummyClassifier(strategy='most_frequent')  
    dummy_model.fit(features_train, target_train)

    # Hacer predicciones con el modelo Dummy
    dummy_predictions = dummy_model.predict(features_valid)

    sklearn_metrics(target_valid,dummy_predictions)


### Función para probar con regresión lineal

def LogisticRegMod(features_train, features_valid, target_train, target_valid):

    print("Modelo de regresión Logística:")
    model = LogisticRegression()
    model.fit(features_train, target_train)
    predictions_valid = model.predict(features_valid)

    sklearn_metrics(target_valid, predictions_valid)

    recall_presicion_roc(model,features_valid, target_valid)
    
   #return predictions_valid, model





def grid_search_decision_tree(X_train, X_valid, y_train, y_valid, param_grid):
    #Realiza una búsqueda de cuadrícula para encontrar los mejores parámetros para un árbol de decisión."""
    # Inicializar el modelo de árbol de decisión
    dt_classifier = DecisionTreeClassifier()
    
    # Inicializar el objeto GridSearchCV
    grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5)
    
    # Realizar la búsqueda de cuadrícula en los datos de entrenamiento
    grid_search.fit(X_train, y_train)
    
    # Obtener los mejores parámetros y el mejor modelo
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Evaluar el mejor modelo en los datos de validación
    pred_DT_GS = best_model.predict(X_valid)

    sklearn_metrics(y_valid, pred_DT_GS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)



def grid_search_random_forest(X_train, X_valid, y_train, y_valid, param_grid):
    ### Realiza una búsqueda de cuadrícula para encontrar los mejores parámetros para un Bosque Aleatorio."""
    # Inicializar el modelo de Bosque Aleatorio
    rf_classifier = RandomForestClassifier()
    
    # Inicializar el objeto GridSearchCV
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
    
    # Crear barra de progreso
    with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])) as pbar:
        # Realizar la búsqueda de cuadrícula en los datos de entrenamiento
        grid_search.fit(X_train, y_train)
        
        # Actualizar la barra de progreso
        pbar.update() 
   
    
    # Obtener los mejores parámetros y el mejor modelo
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    pred_RF_GS = best_model.predict(X_valid)

    sklearn_metrics(y_valid, pred_RF_GS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)

    
    # Evaluar el mejor modelo en los datos de validación
    #accuracy = best_model.score(X_valid, y_valid)


def random_search_random_forest(X_train, X_valid, y_train, y_valid, param_dist, n_iter=100):
    #Realiza una búsqueda aleatoria para encontrar los mejores parámetros para un Bosque Aleatorio."""
    # Inicializar el modelo de Bosque Aleatorio
    rf_classifier = RandomForestClassifier()

    # Inicializar el objeto RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=n_iter, cv=5, random_state=42, n_jobs=-1)
    
    # Realizar la búsqueda aleatoria en los datos de entrenamiento
    random_search.fit(X_train, y_train)
    
    # Obtener los mejores parámetros y el mejor modelo
    best_params = random_search.best_params_
    print("Mejores parámetros encontrados:", best_params)

    best_model = random_search.best_estimator_

    pred_RF_RS = best_model.predict(X_valid)

    sklearn_metrics(y_valid, pred_RF_RS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)
    
    # Evaluar el mejor modelo en los datos de validación
    #accuracy = best_model.score(X_valid, y_valid)


def random_search_lgbm(X_train, X_valid, y_train, y_valid, param_dist, n_iter=100):
    #"""Realiza una búsqueda aleatoria para encontrar los mejores parámetros para LightGBM."""
    # Inicializar el modelo LightGBM

        #Configurando LightGB para que solo muestre errores:
    lgbm_params = {
        'verbosity': -1,
        'n_jobs': -1
    }
    
    lgb_classifier = lgb.LGBMClassifier(**lgbm_params)

    # Inicializar el objeto RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=lgb_classifier, param_distributions=param_dist, n_iter=n_iter, cv=5, random_state=42, n_jobs=-1)

    # Realizar la búsqueda aleatoria en los datos de entrenamiento
    random_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor modelo
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    # Evaluar el mejor modelo en los datos de validación
    y_pred = best_model.predict(X_valid)
    #accuracy = accuracy_score(y_valid, y_pred)

    pred_LGBM_RS = best_model.predict(X_valid)

    sklearn_metrics(y_valid, pred_LGBM_RS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)


    #return best_params, accuracy, best_model




def grid_search_lgbm(X_train, X_valid, y_train, y_valid, param_grid, n_iter=100):

    lgbm_params = {
        'verbosity': -1,
        'n_jobs': -1
    }

    model_lgbm_gs = lgb.LGBMClassifier(random_state=42, **lgbm_params)


    # Configurar GridSearchCV
    grid_search = GridSearchCV(estimator=model_lgbm_gs, param_grid=param_grid, 
                               scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

    # Realizar la búsqueda de los mejores hiperparámetros
    grid_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Entrenar el modelo con los mejores parámetros en el conjunto de entrenamiento completo
    best_model = grid_search.best_estimator_


    # Calcular la precisión
    #accuracy = accuracy_score(y_valid, predictions)

    # Predecir en el conjunto de validación
    pred_LGBM_GS = best_model.predict(X_valid)

    sklearn_metrics(y_valid, pred_LGBM_GS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)


# Definir el modelo
def grid_search_catboost(X_train, X_valid, y_train, y_valid, param_grid):
    
    model_catboost = cb.CatBoostClassifier(random_state=42, silent=True)

    # Configurar GridSearchCV
    grid_search = GridSearchCV(estimator=model_catboost, param_grid=param_grid, 
                               scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

    # Realizar la búsqueda de los mejores hiperparámetros
    grid_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Entrenar el modelo con los mejores parámetros en el conjunto de entrenamiento completo
    best_model = grid_search.best_estimator_

    
    #predictions = best_model.predict(X_valid)

    # Calcular la precisión
    #accuracy = accuracy_score(target_valid, predictions)

    # Predecir en el conjunto de validación
    pred_CB_GS = best_model.predict(X_valid)

    sklearn_metrics(y_valid, pred_CB_GS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)



def random_search_catboost(X_train, X_valid, y_train, y_valid, param_dist, n_iter=100):
    #"""Realiza una búsqueda aleatoria para encontrar los mejores parámetros para CatBoost."""
    # Inicializar el modelo CatBoost
    catboost_classifier = cb.CatBoostClassifier(verbose=0)

    # Inicializar el objeto RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=catboost_classifier, param_distributions=param_dist, n_iter=n_iter, cv=5, random_state=42, n_jobs=-1)

    # Realizar la búsqueda aleatoria en los datos de entrenamiento
    random_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor modelo
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    # Evaluar el mejor modelo en los datos de validación
    pred_CB_RS = best_model.predict(X_valid)
    #accuracy = accuracy_score(y_valid, y_pred)

    sklearn_metrics(y_valid, pred_CB_RS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)


## Demora aproximadamente 30 min en correr.
def random_search_xgboost(X_train, X_valid, y_train, y_valid, param_dist, n_iter=100):
    #"""Realiza una búsqueda aleatoria para encontrar los mejores parámetros para XGBoost."""

    # Convertir los valores de True/False a 1/0 en target_train y target_valid
    y_train = y_train.astype(int)
    y_valid = y_valid.astype(int)

    # Inicializar el modelo XGBoost
    xgboost_classifier = xgb.XGBClassifier(eval_metric='logloss')

    # Inicializar el objeto RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=xgboost_classifier, param_distributions=param_dist, n_iter=n_iter, cv=5, random_state=42, n_jobs=-1)

    # Realizar la búsqueda aleatoria en los datos de entrenamiento
    random_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor modelo
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    # Evaluar el mejor modelo en los datos de validación
    pred_XGB_RS = best_model.predict(X_valid)

    # accuracy = accuracy_score(y_valid, y_pred)

    sklearn_metrics(y_valid, pred_XGB_RS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)


# Definir el modelo
def grid_search_xgboost(X_train, X_valid, y_train, y_valid, param_grid):
    
    # Definir el modelo
    model_xgb = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                  random_state=42, use_label_encoder=False)
    # Configurar GridSearchCV
    grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, 
                               scoring='accuracy', cv=5, verbose=2, n_jobs=-1)

    # Realizar la búsqueda de los mejores hiperparámetros
    grid_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Entrenar el modelo con los mejores parámetros en el conjunto de entrenamiento completo
    best_model = grid_search.best_estimator_

    # Predecir en el conjunto de validación
    pred_XGB_GS = best_model.predict(X_valid)

    # Calcular la precisión
    #accuracy = accuracy_score(target_valid, predictions)

    sklearn_metrics(y_valid, pred_XGB_GS)
    
    recall_presicion_roc(best_model, X_valid, y_valid)