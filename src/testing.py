from evaluate import datasets_paths, scorer, standard_scaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from c45 import C45
from xgboost import XGBClassifier
from sklearn.svm import SVC
from preprocessing import preprocess
from scipy.stats import ttest_rel
from tabulate import tabulate
from time import time
from colorama import Fore
from pathlib import Path

import numpy as np

# lung_cancer, diabetes, credit_cards
datasets = datasets_paths()[:2]
classifiers_types = {
    'C45': C45(),
    'SVM': SVC(),
    'XGB': XGBClassifier()
}
techniques = {
    'NONE': None,
    'SMOTE': SMOTE(),
    'RND_OVER': RandomOverSampler(),
    'RND_UNDER': RandomUnderSampler(),
}
pipelines = []
models_description = []
for classifier in classifiers_types.items():
    for technique in techniques.items():
        steps = [(f"{technique[0]}", technique[1]), ('model', classifier[1])]
        models_description.append(f"{classifier[0]} {technique[0]}")
        pipelines.append(Pipeline(steps=steps))


def conduct_test(n_splits: int, n_repeats: int, file_name: str):
    repeated_stratified_k_fold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

    scores = np.zeros((len(pipelines), len(datasets), n_splits * n_repeats))

    for dataset_id, dataset in enumerate(datasets):
        measurement_start = time()
        print(f"{Fore.YELLOW} Testing: {dataset}")
        dataset_name = Path(dataset).name
        contains_categorical_col = True if dataset_id == 0 else False
        X, y = preprocess(dataset, 0 if contains_categorical_col else None)
        for fold_id, (train, test) in enumerate(repeated_stratified_k_fold.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            X_train, sc = standard_scaler(X_train, contains_categorical_col)
            if contains_categorical_col:
                X_test = sc.transform(X=X_test[:, 2:])
            else:
                X_test = sc.transform(X=X_test)

            for pipeline_id, pipeline in enumerate(pipelines):
                pipeline = clone(pipeline)
                pipeline.fit(X_train, y_train)
                scores[pipeline_id, dataset_id, fold_id] = scorer(pipeline, X_test, y_test)
        measurement_stop = time()
        execution_time = measurement_stop - measurement_start
        print(f"{Fore.GREEN} {dataset_name} took: {execution_time} seconds")

    np.save(file_name, scores)
    print(f"{Fore.RESET}")  # reset colors


def analyze_results(result_file_name: str) -> None:
    scores = np.load(f"{result_file_name}.npy")
    print(f"Scores: {scores.shape}")


def perform_statistics(scores: np.array, alpha: float) -> str:
    t_statistics = np.zeros((len(pipelines), len(pipelines)))
    p_value = np.zeros((len(pipelines), len(pipelines)))
    for i in range(len(pipelines)):
        for j in range(len(pipelines)):
            for k in range(len(datasets)):
                t_statistics[i, j], p_value[i, j] = ttest_rel(scores[i][k], scores[j][k])

    t_statistics = np.nan_to_num(t_statistics)
    p_value = np.nan_to_num(p_value, nan=1.0)

    headers = models_description
    column_names = np.array(list(map(lambda name: [name], headers)))
    t_statistic_table = np.concatenate((column_names, t_statistics), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((column_names, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print(f"t-statistic:\n{t_statistic_table}\n\np-value:\n{p_value_table}\n\n")

    advantage = np.zeros((len(pipelines), len(pipelines)))
    advantage[t_statistics > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (column_names, advantage), axis=1)
        , headers)
    print(f"Advantage:\n{advantage_table}\n\n")

    significance = np.zeros((len(pipelines), len(pipelines)))
    significance[p_value <= alpha] = 1
    significance_table = tabulate(np.concatenate(
        (column_names, significance), axis=1
    ), headers)
    print(f"Statistical significance (alpha = {alpha}):\n{significance_table}\n\n")

    statistically_better = significance * advantage
    statistically_better_table = tabulate(np.concatenate(
        (column_names, statistically_better), axis=1
    ), headers)
    print(f"Statistically significantly better:\n{statistically_better_table}")


if __name__ == "__main__":
    # conduct_test(2, 5, "results")
    analyze_results("results")
