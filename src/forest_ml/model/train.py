import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
import click

from sklearn.metrics import log_loss, f1_score
from sqlalchemy import false

from forest_ml.visualization.profiling import create_report_folder

from .lr_model import create_logistic_reg_pipeline
from .knn_models import create_KNN_pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from ..data_prep.train_val import make_split

METRICS = ['f1_micro', 'f1_macro', 'f1_weighted', 'roc_auc_ovr', 'roc_auc_ovo','neg_log_loss']

def save_model(model, path:str = None, name_prefix:str=None):
    if path==None:
        path = os.path.abspath('models')
        if os.path.exists(path) == False:
            os.mkdir(path)
    joblib.dump(model, os.path.join(path, f"{name_prefix}.pkl"))

    return os.path.join(path, f"{name_prefix}.pkl")

def load_model(path):
    return joblib.load(path)
    

def train_lr (
    data_path:str,
    name_prefix:str,
    save_model_path:str=None,
    use_scaller: bool = True,
    use_PCA: int = 0,
    max_iter: int = 100,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'lbfgs'

    
)->None:
    with mlflow.start_run():
        data = make_split(data_path=data_path)
        # metrics
        metrics = METRICS
        # cv strategy
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # model
        model = create_logistic_reg_pipeline(
            use_scaller=use_scaller,
            use_PCA=use_PCA,
            max_iter=max_iter,
            C=C,
            penalty=penalty,
            solver=solver
        )
        # cross val
        cv_results = cross_validate(
            model, data['X_train'], data['y_train'], cv=cv, scoring=metrics,
            return_estimator=False, error_score='raise'
        )
        cv_results = pd.DataFrame(cv_results).sort_values(['test_neg_log_loss'], ascending=False)

        model.fit(data['X_train'], data['y_train'])
        pred = model.predict(data['X_test'])
        log_loss_test = log_loss(data['y_test'], model.predict_proba(data['X_test']), labels=data['y_test'])
        f1_score_test = f1_score(data['y_test'], pred, average='micro')
        
        path_saved_model = save_model(model, name_prefix=name_prefix, path=save_model_path)
        
        click.echo(f'Model saved to:')
        click.echo(os.path.abspath(path_saved_model))
        
        # return cv_results
        # add model params to mlflow
        mlflow.log_param('model_name_prefix', name_prefix)
        mlflow.log_param('use_scaller', use_scaller)
        mlflow.log_param('use_pca_with_n_components', use_PCA)
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


def train_knn (
    data_path:str,
    name_prefix:str,
    save_model_path:str=None,
    use_scaller: bool = True,
    use_PCA: int = 0,
    n_neighbors:int = 5,
    weights:str = 'uniform',
    algorithm:str = 'auto',
    leaf_size:int = 30,
    p:int = 2,
    metric:str = 'minkowski'
)->None:

    with mlflow.start_run():
        data = make_split(data_path=data_path)
        # metrics
        metrics = METRICS
        # cv strategy
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        model = create_KNN_pipeline(
            use_scaller=use_scaller,
            use_pca=use_PCA,
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric
        )

        cv_results = cross_validate(
            model, data['X_train'], data['y_train'], cv=cv, scoring=metrics,
            return_estimator=False, error_score='raise'
        )
        cv_results = pd.DataFrame(cv_results).sort_values(['test_neg_log_loss'], ascending=False)

        model.fit(data['X_train'], data['y_train'])
        pred = model.predict(data['X_test'])
        log_loss_test = log_loss(data['y_test'], model.predict_proba(data['X_test']), labels=data['y_test'])
        f1_score_test = f1_score(data['y_test'], pred, average='micro')
        
        path_saved_model = save_model(model, name_prefix=name_prefix, path=save_model_path)
        
        click.echo(f'Model saved to:')
        click.echo(os.path.abspath(path_saved_model))
        # return cv_results
        # add model params to mlflow
        mlflow.log_param('model_name_prefix', name_prefix)
        mlflow.log_param('use_scaller', use_scaller)
        mlflow.log_param('use_pca_with_n_components', use_PCA)
        mlflow.log_param('n_neighbors', n_neighbors)
        mlflow.log_param('weights', weights)
        mlflow.log_param('algorithm', algorithm)
        mlflow.log_param('leaf_size', leaf_size)
        mlflow.log_param('p', p)
        mlflow.log_param('metric', metric)
        # add metrics for tracking
        mlflow.log_metric('val_log_loss_cv_mean', cv_results['test_neg_log_loss'].mean())
        mlflow.log_metric('val_f1_micro_cv_mean', cv_results['test_f1_micro'].mean())
        mlflow.log_metric('val_f1_macro_cv_mean', cv_results['test_f1_macro'].mean())
        mlflow.log_metric('val_f1_weighted_cv_mean', cv_results['test_f1_weighted'].mean())
        mlflow.log_metric('val_roc_auc_ovr_cv_mean', cv_results['test_roc_auc_ovr'].mean())
        mlflow.log_metric('val_roc_auc_ovo_cv_mean', cv_results['test_roc_auc_ovo'].mean())
        mlflow.log_metric('test_log_loss', log_loss_test)
        mlflow.log_metric('test_f1_micro', f1_score_test)


def train_lr_autotuned (
    data_path:str,
    name_prefix:str,
    save_model_path:str=None,
    use_scaller: bool = True,
    use_PCA: int = 0,
    
)->None:
    """
    Grid searching the key hyperparameters for LogisticRegression
    """
    with mlflow.start_run():
        data = make_split(data_path=data_path)
        # metrics
        metrics = METRICS
        # grid
        grid = {
            'clf__solver':['liblinear','saga'],
            'clf__penalty':['l1', 'l2'],
            'clf__C':[0.1, 100, 10, 0.1, 0.01]
        }
        # cv strategy
        cv_inner = KFold(n_splits=10, random_state=42, shuffle=True)
        cv_outter = KFold(n_splits=5, random_state=42, shuffle=True)
        # model
        model = create_logistic_reg_pipeline(use_scaller=use_scaller, use_PCA=use_PCA, max_iter=10000)
        # search
        search = GridSearchCV(model, grid, scoring=METRICS[-1], n_jobs=1, cv=cv_inner, refit=True, verbose=0)
        # cross val
        cv_results = cross_validate(
            search, data['X_train'], data['y_train'], cv=cv_outter, scoring=metrics,
            return_estimator=True, error_score='raise', verbose=1, n_jobs=-1,
        )
        cv_results = pd.DataFrame(cv_results).sort_values(['test_neg_log_loss'], ascending=False)

        # return cv_results
        model  = cv_results['estimator'].iloc[0]
        # model.fit(data['X_train'], data['y_train'])
        pred = model.predict(data['X_test'])
        log_loss_test = log_loss(data['y_test'], model.predict_proba(data['X_test']), labels=data['y_test'])
        f1_score_test = f1_score(data['y_test'], pred, average='micro')
        
        path_saved_model = save_model(model, name_prefix=name_prefix, path=save_model_path)
        
        click.echo(f'Model saved to:')
        click.echo(os.path.abspath(path_saved_model))
        
        model_params = model.best_estimator_.get_params()
        # return cv_results
        # add model params to mlflow
        mlflow.log_param('model_name_prefix', name_prefix)
        mlflow.log_param('use_scaller', use_scaller)
        mlflow.log_param('use_pca_with_n_components', use_PCA)
        mlflow.log_param('max_iter', model_params['clf__max_iter'])
        mlflow.log_param('C', model_params['clf__C'])
        mlflow.log_param('penalty', model_params['clf__penalty'])
        mlflow.log_param('solver', model_params['clf__solver'])
        # add metrics for tracking
        mlflow.log_metric('val_log_loss_cv_mean', cv_results['test_neg_log_loss'].mean())
        mlflow.log_metric('val_f1_micro_cv_mean', cv_results['test_f1_micro'].mean())
        mlflow.log_metric('val_f1_macro_cv_mean', cv_results['test_f1_macro'].mean())
        mlflow.log_metric('val_f1_weighted_cv_mean', cv_results['test_f1_weighted'].mean())
        mlflow.log_metric('val_roc_auc_ovr_cv_mean', cv_results['test_roc_auc_ovr'].mean())
        mlflow.log_metric('val_roc_auc_ovo_cv_mean', cv_results['test_roc_auc_ovo'].mean())
        mlflow.log_metric('test_log_loss', log_loss_test)
        mlflow.log_metric('test_f1_micro', f1_score_test)


def train_knn_autotune(
    data_path:str,
    name_prefix:str,
    save_model_path:str=None,
    use_scaller: bool = True,
    use_PCA: int = 0,
    algorithm:str = 'auto',
    leaf_size:int = 30,
    p:int = 2,
):
    with mlflow.start_run():
        data = make_split(data_path=data_path)
        # metrics
        metrics = METRICS
        # grid
        grid = {
            'clf__weights':['uniform', 'distance'],
            'clf__metric':['euclidean', 'manhattan', 'minkowski'],
            'clf__n_neighbors':range(1, 21, 2)
        }
        # cv strategy
        cv_inner = KFold(n_splits=10, random_state=42, shuffle=True)
        cv_outter = KFold(n_splits=5, random_state=42, shuffle=True)
        # model
        model = create_KNN_pipeline(use_scaller=use_scaller, use_pca=use_PCA, algorithm=algorithm, leaf_size=leaf_size, p=p,)
        # search
        search = GridSearchCV(model, grid, scoring=METRICS[-1], n_jobs=1, cv=cv_inner, refit=True, verbose=0)
        # cross val
        cv_results = cross_validate(
            search, data['X_train'], data['y_train'], cv=cv_outter, scoring=metrics,
            return_estimator=True, error_score='raise', verbose=1, n_jobs=-1,
        )
        cv_results = pd.DataFrame(cv_results).sort_values(['test_neg_log_loss'], ascending=False)

        # return cv_results
        model  = cv_results['estimator'].iloc[0]
        # model.fit(data['X_train'], data['y_train'])
        pred = model.predict(data['X_test'])
        log_loss_test = log_loss(data['y_test'], model.predict_proba(data['X_test']), labels=data['y_test'])
        f1_score_test = f1_score(data['y_test'], pred, average='micro')
        
        path_saved_model = save_model(model, name_prefix=name_prefix, path=save_model_path)
        
        click.echo(f'Model saved to:')
        click.echo(os.path.abspath(path_saved_model))
        
        model_params = model.best_estimator_.get_params()
        # return cv_results
        # add model params to mlflow
        mlflow.log_param('model_name_prefix', name_prefix)
        mlflow.log_param('use_scaller', use_scaller)
        mlflow.log_param('use_pca_with_n_components', use_PCA)
        mlflow.log_param('n_neighbors', model_params['clf__n_neighbors'])
        mlflow.log_param('weights', model_params['clf__weights'])
        mlflow.log_param('algorithm', model_params['clf__algorithm'])
        mlflow.log_param('leaf_size', model_params['clf__leaf_size'])
        mlflow.log_param('p', model_params['clf__p'])
        mlflow.log_param('metric',model_params['clf__metric'])
        # add metrics for tracking
        mlflow.log_metric('val_log_loss_cv_mean', cv_results['test_neg_log_loss'].mean())
        mlflow.log_metric('val_f1_micro_cv_mean', cv_results['test_f1_micro'].mean())
        mlflow.log_metric('val_f1_macro_cv_mean', cv_results['test_f1_macro'].mean())
        mlflow.log_metric('val_f1_weighted_cv_mean', cv_results['test_f1_weighted'].mean())
        mlflow.log_metric('val_roc_auc_ovr_cv_mean', cv_results['test_roc_auc_ovr'].mean())
        mlflow.log_metric('val_roc_auc_ovo_cv_mean', cv_results['test_roc_auc_ovo'].mean())
        mlflow.log_metric('test_log_loss', log_loss_test)
        mlflow.log_metric('test_f1_micro', f1_score_test)

