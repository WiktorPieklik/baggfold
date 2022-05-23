from pathlib import Path
from typing import Tuple

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hmeasure import h_score
from xgboost import XGBClassifier
from c45 import C45
from numpy import array as np_array, mean


def datasets_paths() -> Tuple[str, str, str]:
    src_dir = Path(__file__).parent
    datasets_dir = src_dir / "datasets"
    lung_cancer = str(
        (datasets_dir / "lung_cancer.csv").resolve()
    )
    diabetes = str(
        (datasets_dir / "diabetes.csv").resolve()
    )
    credit_cards = str(
        (datasets_dir / "credit_card.csv").resolve()
    )

    return lung_cancer, diabetes, credit_cards


def scorer(estimator, X, y):
    predictions = np_array([estimator.predict(X)]).T
    n1, n0 = y.sum(), y.shape[0] - y.sum()

    return h_score(y, predictions, severity_ratio=(n1 / n0))


def evaluate_model(steps, X, y, model_description) -> None:
    # define pipeline
    pipeline = Pipeline(steps=steps)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring=scorer, cv=cv, n_jobs=-1)

    print(f"model: {model_description[0]}-{model_description[1]}\t h-score: {mean(scores)}")


def evaluate_reference_models(X: np_array, y: np_array) -> None:
    # define models
    models = [('C45', C45()), ('SVM', SVC()), ('XGB', XGBClassifier())]

    # define oversampling techniques
    oversampling = [('NONE', None), ('SMOTE', SMOTE()), ('rnd_over', RandomOverSampler())]

    # define undersampling techniques
    undersampling = [('rnd_under', RandomUnderSampler())]

    for model in models:
        for over in oversampling:
            steps = [('over', over[1]), ('model', model[1])]
            model_description = (model[0], over[0])
            evaluate_model(steps=steps, X=X, y=y, model_description=model_description)

        for under in undersampling:
            steps = [('under', under[1]), ('model', model[1])]
            model_description = (model[0], under[0])
            evaluate_model(steps=steps, X=X, y=y, model_description=model_description)
