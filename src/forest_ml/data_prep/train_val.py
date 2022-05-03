import os
from numpy import save
import pandas as pd
from sklearn.model_selection import train_test_split


# RAW_PATH = os.path.join(os.getcwd())# ,'data', 'raw', 'train.csv')
# PROCESSED_DIR_PATH = os.path.join('..','data', 'processed')

def make_split(data_path, random_seed:int = 42, val_partition:int = 0.2, save_split:bool=True)->dict:
    """
    Make split of training data into train, validation and test parts
    data_path  - path to data
    val_partition - validation part
    save_split - if True make train_test_split with stratified = y
    return dict
    {
        'X_train': X_train,
        'X_val':X_val,
        'X_test':X_test,
        'y_train':y_train,
        'y_val':y_val,
        'y_test':y_test,
    }
    """
    df_raw = pd.read_csv(data_path, index_col='Id')
    
    X = df_raw.drop(['Cover_Type'], axis=1)
    y = df_raw['Cover_Type']    

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, stratify=y_test_val, test_size=0.5)

    return {
        'X_train': X_train,
        'X_val':X_val,
        'X_test':X_test,
        'y_train':y_train,
        'y_val':y_val,
        'y_test':y_test,
    }


