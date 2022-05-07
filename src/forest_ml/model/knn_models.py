from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_KNN_pipeline(
    use_scaller:bool = True,
    use_pca:int = 0,
    n_neighbors:int = 5,
    weights:str = 'uniform',
    algorithm:str = 'auto',
    leaf_size:int = 30,
    p:int = 2,
    metric:str = 'minkowski'
)->Pipeline:
    """
    Create pipeline for KNN model
    """
    pipeline_steps = []

    if use_scaller:
        pipeline_steps.append(
            ('std_scl', StandardScaler())
        )
    
    if use_pca>0:
        pipeline_steps.append(
            ('pca', PCA(n_components=use_pca))
        )
    
    pipeline_steps.append(
        ('clf', KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric
        ))
    )
    return Pipeline