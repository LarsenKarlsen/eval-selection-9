import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from .lr_model import create_logistic_reg_pipeline
from sklearn.model_selection import KFold, cross_validate
from ..data_prep.train_val import make_split

def save_model(model, path:str = None, name_prefix:str=None):
    if path==None:
        path = os.path.abspath('../models')
        if os.path.exists(path) == False:
            os.mkdir(path)
        joblib.dump(model, os.path.join(path, f"{name_prefix}.pkl"))

def load_model(path):
    return joblib.load(path)
    

def train_lr (
    data_path:str,
    save_model_path:str,
    
)->None:
    data = make_split(data_path=data_path)
    # metrics
    metrics = ['f1_micro', 'f1_macro', 'f1_weighted', 'roc_auc_ovr', 'roc_auc_ovo','neg_log_loss']
    # cv strategy
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    # model
    model = create_logistic_reg_pipeline()
    # cross val
    cv_results = cross_validate(
        model, data['X_train'], data['y_train'], cv=cv, scoring=metrics,
        return_estimator=True, error_score='raise'
    )
    cv_results = pd.DataFrame(cv_results).sort_values(['test_neg_log_loss'], ascending=False)

    # joblib.dump()
    return cv_results.iloc[0]
