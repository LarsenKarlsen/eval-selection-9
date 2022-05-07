from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def create_logistic_reg_pipeline (
    use_scaller: bool = True,
    use_PCA: int = 0,
    max_iter: int = 100,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'lbfgs'
)->Pipeline:
    """
    Creates pipeline for logreg function
    """
    pipeline_steps = []

    if use_scaller:
        pipeline_steps.append(
            ('std_scl', StandardScaler())
        )
    
    if use_PCA>0:
        pipeline_steps.append(
            ('pca', PCA(n_components=use_PCA))
        )

    pipeline_steps.append(
        ('clf', LogisticRegression(
            max_iter=max_iter,
            C=C,
            penalty=penalty,
            solver=solver
            ))
    )

    return Pipeline(steps=pipeline_steps)