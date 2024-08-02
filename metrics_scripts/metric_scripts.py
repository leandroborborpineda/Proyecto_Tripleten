## Librería para métricas de las predicciones:
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def sklearn_metrics(target_valid,  predictions):
    
    print("Exactitud del modelo: ", accuracy_score(target_valid, predictions))

    #print("Exactitud del modelo:", accuracy)

    print("Matrix de confusión: \n",confusion_matrix(target_valid, predictions))

    print("Precisión del modelo: ", precision_score(target_valid, predictions))

    print("Calificación de F1: ", f1_score(target_valid, predictions))

    print("AUC-ROC del modelo: ", roc_auc_score(target_valid, predictions))


### Funcion para tabla Precision - Recall:

def precision_recall_table(probabilities, target_valid):

    for threshold in np.arange(0, 1.0 , 0.02):
        predicted_valid = probabilities > threshold
        precision = precision_score(target_valid, predicted_valid)
        recall = recall_score(target_valid, predicted_valid)

        print('Threshold = {:.2f} | Precision = {:.3f}, Recall = {:.3f}'.format(threshold, precision, recall))


def prec_recall_curve(target_valid, probabilities):
    
    precision_c, recall_c, thresholds = precision_recall_curve(target_valid, probabilities)
    
    plt.figure(figsize=(6, 6))
    plt.step(recall_c, precision_c, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.show() 


### CUrva ROC

def roc_curve_func(target_valid_f, probabilities_one_valid_f):
    fpr, tpr, thresholds = roc_curve(target_valid_f, probabilities_one_valid_f)

    plt.figure()
    plt.plot(fpr, tpr)

    # Curva ROC para modelo aleatorio 
    plt.plot([0, 1], [0, 1], linestyle='--')

    # < utiliza las funciones plt.xlim() y plt.ylim() para
    #   establecer el límite para los ejes de 0 a 1 >

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # < utiliza las funciones plt.xlabel() y plt.ylabel() para
    #   nombrar los ejes "Tasa de falsos positivos" y "Tasa de verdaderos positivos">
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')

    # < agrega el encabezado "Curva ROC" con la función plt.title() >
    plt.title('Curva ROC')


    plt.show()