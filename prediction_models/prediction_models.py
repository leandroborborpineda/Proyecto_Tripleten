## Librerias para entrenamiento:
## Modelo Regresión Logística
from sklearn.linear_model import LogisticRegression
## Modelo Decision Tree
from sklearn.tree import DecisionTreeClassifier
## Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
## Modelo Dummy
from sklearn.dummy import DummyClassifier


#from data.data_split_train_test import data_split
from metrics_scripts.metric_scripts import sklearn_metrics

## Light GBM
import lightgbm as lgb
from scipy.stats import randint, uniform

##Catboost
import catboost as cb

## XGBoost
import xgboost as xgb


def dummy_model_script(features_train, features_valid, target_train, target_valid):
    
    dummy_model = DummyClassifier(strategy='most_frequent')  
    dummy_model.fit(features_train, target_train)

    # Hacer predicciones con el modelo Dummy
    dummy_predictions = dummy_model.predict(features_valid)

    sklearn_metrics(target_valid,dummy_predictions)


### Función para probar con regresión lineal

def LogisticRegMod(features_train, features_valid, target_train, target_valid):
    model = LogisticRegression()
    model.fit(features_train, target_train)
    predictions_valid = model.predict(features_valid)
          
    sklearn_metrics(target_valid, predictions_valid)

    
        
    
       
    return predictions_valid, model