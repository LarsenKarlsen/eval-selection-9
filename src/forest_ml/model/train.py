import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
import click

from sklearn.metrics import log_loss, f1_score

from .lr_model import create_logistic_reg_pipeline
from sklearn.model_selection import KFold, cross_validate
from ..data_prep.train_val import make_split

def save_model(model, path:str = None, name_prefix:str=None):
    if path==None:
        path = os.path.abspath('../models')
        if os.path.exists(path) == False:
            os.mkdir(path)
        joblib.dump(model, os.path.join(path, f"{name_prefix}.pkl"))
    return os.path.join(path, f"{name_prefix}.pkl")

def load_model(path):
    return joblib.load(path)
    

@click.command()
@click.option(
    "-d",
    "--data-path",
    default="data/heart.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=click.Path),
    show_default=True,
)
@click.option(
    "--model-name-prefix",
    default="LR_model",
    type=str,
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models/",
    type=click.Path(dir_okay=False, writable=True, path_type=click.Path),
    show_default=True,
)
@click.option(
    "--use-scaller",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--C",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--penalty",
    default='l2',
    type=str,
    show_default=True,
)
@click.option(
    "--solver",
    default='lbfgs',
    type=str,
    show_default=True,
)
def train_lr (
    data_path:str,
    name_prefix:str,
    save_model_path:str=None,
    use_scaller: bool = True,
    max_iter: int = 100,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'lbfgs'

    
)->None:
    with mlflow.start_run():
        data = make_split(data_path=data_path)
        # metrics
        metrics = ['f1_micro', 'f1_macro', 'f1_weighted', 'roc_auc_ovr', 'roc_auc_ovo','neg_log_loss']
        # cv strategy
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # model
        model = create_logistic_reg_pipeline(use_scaller=use_scaller)
        # cross val
        cv_results = cross_validate(
            model, data['X_train'], data['y_train'], cv=cv, scoring=metrics,
            return_estimator=True, error_score='raise'
        )
        cv_results = pd.DataFrame(cv_results).sort_values(['test_neg_log_loss'], ascending=False)

        model.fit(data['X_train'], data['y_train'])
        pred = model.predict(data['X_test'])
        log_loss_test = log_loss(data['y_test'], model.predict_proba(data['X_test']), labels=data['y_test'])
        f1_score_test = f1_score(data['y_test'], pred, average='micro')
        
        path_saved_model = save_model(model, name_prefix=name_prefix, path=save_model_path)
        
        print(f'Model saved to {path_saved_model}')
        
        # return cv_results
        # add model params to mlflow
        mlflow.log_param('model_name_prefix', name_prefix),
        mlflow.log_param('use_scaller', use_scaller)
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_param('C', C)
        mlflow.log_param('penalty', penalty)
        mlflow.log_param('solver', solver)
        # add metrics for tracking
        mlflow.log_metric('val_log_loss_cv_mean', cv_results['test_neg_log_loss'].mean())
        mlflow.log_metric('val_f1_micro_cv_mean', cv_results['test_f1_micro'].mean())
        mlflow.log_metric('val_f1_macro_cv_mean', cv_results['test_f1_macro'].mean())
        mlflow.log_metric('val_f1_weighted_cv_mean', cv_results['test_f1_weighted'].mean())
        mlflow.log_metric('val_roc_auc_ovr_cv_mean', cv_results['test_roc_auc_ovr'].mean())
        mlflow.log_metric('val_roc_auc_ovo_cv_mean', cv_results['test_roc_auc_ovo'].mean())
        mlflow.log_metric('test_log_loss', log_loss_test)
        mlflow.log_metric('test_f1_micro', f1_score_test)



        


