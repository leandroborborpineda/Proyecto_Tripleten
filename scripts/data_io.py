import pandas as pd
import os

def save_dataframe(df, filepath, format='csv'):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError("Format not supported. Please use 'csv' or 'pickle'.")

def load_dataframe(filepath, format='csv'):
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError("Format not supported. Please use 'csv' or 'pickle'.")