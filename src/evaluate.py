from pathlib import Path
from typing import Tuple

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hmeasure import h_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import array as np_array, mean
from promting import INFO, SUCCESS


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


def repeated_cv(X: np_array, y: np_array, pipeline: Pipeline, k_folds: int, repeats: int,
                contains_categorical_col: bool) -> list:
    repeated_stratified_k_fold = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=repeats, random_state=1)
    scores = []
    for train_index, test_index in repeated_stratified_k_fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, sc = standard_scaler(X_train, contains_categorical_col)
        if contains_categorical_col:
            X_test = sc.transform(X=X_test[:, 2:])
        else:
            X_test = sc.transform(X=X_test)
        pipeline.fit(X_train, y_train)
        score = scorer(pipeline, X_test, y_test)
        scores.append(score)

    return scores


def standard_scaler(X: np_array, contains_categorical_col: bool) -> Tuple[np_array, StandardScaler]:
    sc = StandardScaler()
    x = None
    if contains_categorical_col:
        x = sc.fit_transform(X=X[:, 2:])
    else:
        x = sc.fit_transform(X=X)

    return x, sc


def evaluate_model(steps, X, y, model_description, contains_categorical_col: bool = False,
                   save_result: bool = False) -> Tuple[str, float]:
    # define pipeline
    pipeline = Pipeline(steps=steps)
    scores = repeated_cv(X=X, y=y, pipeline=pipeline, k_folds=2, repeats=5,
                         contains_categorical_col=contains_categorical_col)

    mean_score = mean(scores)
    full_model_name = f"{model_description[0]}-{model_description[1]}"
    INFO(f"model: {full_model_name}\t h-score: {mean_score}")

    return full_model_name, mean_score


def evaluate_reference_models(X: np_array, y: np_array, contains_categorical_col: bool = False,
                              results_file: str = None) -> None:
    # define models
    models = [('CART', DecisionTreeClassifier()), ('SVM', SVC()), ('XGB', XGBClassifier())]

    # define oversampling techniques
    oversampling = [('NONE', None), ('SMOTE', SMOTE()), ('rnd_over', RandomOverSampler())]

    # define undersampling techniques
    undersampling = [('rnd_under', RandomUnderSampler())]

    if results_file is not None:
        with open(results_file, "w") as file:
            file.write("Model;Score\n")

    for model in models:
        for over in oversampling:
            steps = [('over', over[1]), ('model', model[1])]
            model_description = (model[0], over[0])
            model_name, score = evaluate_model(steps=steps, X=X, y=y, model_description=model_description,
                                               contains_categorical_col=contains_categorical_col)
            if results_file is not None:
                with open(results_file, "a") as file:
                    file.write(f"{model_name};{score}\n")

        for under in undersampling:
            steps = [('under', under[1]), ('model', model[1])]
            model_description = (model[0], under[0])
            model_name, score = evaluate_model(steps=steps, X=X, y=y, model_description=model_description,
                           contains_categorical_col=contains_categorical_col)

            if results_file is not None:
                with open(results_file, "a") as file:
                    file.write(f"{model_name};{score}\n")

    if results_file is not None:
        SUCCESS(f"Results saved in: {results_file}")
