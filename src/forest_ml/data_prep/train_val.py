import os
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_PATH = os.path.join('..','data', 'raw', 'train.csv')
PROCESSED_DIR_PATH = os.path.join('..','data', 'processed')

def make_split(random_seed:int = 42, val_partition:int = 0.2, save_split:bool=True):
    """
    Make split of training data into train and validation parts
    val_partition - validation part
    save_split - if True make train_test_split with stratified = y
    """
    df_raw = pd.read_csv(RAW_PATH, index_col='Id')



