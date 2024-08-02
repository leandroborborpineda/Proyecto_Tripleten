# import pandas as pd
# from functools import reduce
# import os
# import sys
# import math
# from scipy import stats as st
# import time
# import random 
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

## Función para dividir entre datos de entrenamiento y de validación
def data_split(data, split_size):
    target = data['contract_status']
    features = data.drop('contract_status', axis=1)
    
    features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size = split_size,
                                                                                  random_state = 12345)
    
    ##
    # print(features_train.shape)
    # print(features_valid.shape)
    # print(target_train.shape)
    # print(target_valid.shape)

    ##

    return features_train, features_valid, target_train, target_valid




