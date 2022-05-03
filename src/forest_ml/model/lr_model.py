from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def create_logistic_reg_pipeline (
    use_scaller: bool = True,
)->Pipeline:
    """
    Creates pipeline for logreg function
    """
    pipeline_steps = []

    if use_scaller:
        pipeline_steps.append(
            ('std_scl', StandardScaler())
        )

    pipeline_steps.append(
        ('clf', LogisticRegression())
    )

    return Pipeline(steps=pipeline_steps)