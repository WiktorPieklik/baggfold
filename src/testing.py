from __future__ import annotations

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
from colorama import Fore
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from promting import INFO, WARNING, IMPORTANT, SUCCESS
from alive_progress import alive_bar
from baggfold import ThreadedBaggFold

import numpy as np

# lung_cancer, diabetes, credit_cards
# datasets = datasets_paths()[:2]
classifiers_types = {
    # 'C45': C45(),
    'SVM': SVC(),
    'XGB': XGBClassifier(),
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
        steps = [(technique[0], technique[1]), (classifier[0], classifier[1])]
        models_description.append(f"{classifier[0]} {technique[0]}")
        pipelines.append(Pipeline(steps=steps))
pipelines.append(Pipeline(steps=[('NONE', None), ('BaggFold', ThreadedBaggFold(lambda: SVC()))]))


class StatisticalTest:
    def __init__(self, datasets_paths: List[str], pipelines: List[Pipeline], alpha: float, n_splits: int,
                 n_repeats: int):
        self.__datasets = datasets_paths
        self.__pipelines = pipelines
        self.__dt_format = "%d-%m-%Y_%H:%M%S"
        self.__alpha = alpha
        self.__n_splits = n_splits
        self.__n_repeats = n_repeats
        self.__started_at: str | None = None
        self.__scores: dict = {}
        self.__t_statistics: dict = {}
        self.__p_values: dict = {}

    @property
    def __test_type(self) -> str:
        title = "Pair testing using t-Student statistic on"
        tests_len = len(self.__datasets)
        specification = "one dataset" if tests_len == 1 else f"{tests_len} datasets"
        alpha = f"with trust level alpha={self.__alpha}"

        return title + " " + specification + " " + alpha

    def __calculate_statistics(self) -> None:
        repeated_stratified_k_fold = RepeatedStratifiedKFold(n_splits=self.__n_splits, n_repeats=self.__n_repeats)
        WARNING("\n\n[1/2] Calculating statistics...")
        for dataset_path in self.__datasets:
            score = np.zeros((len(self.__pipelines), self.__n_splits * self.__n_repeats))
            dataset_name = self.__get_dataset_name(dataset_path)
            contains_categorical_col = True if "lung_cancer" in dataset_name else False
            X, y = preprocess(dataset_path, 0 if contains_categorical_col else None)
            print(Fore.CYAN, end='')
            with alive_bar(len(self.__pipelines) * score.shape[1], dual_line=True,
                           title=f"Testing {dataset_name}") as bar:
                for fold_id, (train, test) in enumerate(repeated_stratified_k_fold.split(X, y)):
                    X_train, X_test = X[train], X[test]
                    y_train, y_test = y[train], y[test]
                    X_train, sc = standard_scaler(X_train, contains_categorical_col)
                    if contains_categorical_col:
                        X_test = sc.transform(X=X_test[:, 2:])
                    else:
                        X_test = sc.transform(X=X_test)

                    for pipeline_id, pipeline in enumerate(self.__pipelines):
                        classifier_name, preprocessing = self.__pipeline_credentials(pipeline)
                        bar.text(
                            f"->  Classifier: {classifier_name} ({', '.join(preprocessing)}), Fold no. {fold_id + 1}")
                        pipeline = clone(pipeline)
                        pipeline.fit(X_train, y_train)
                        score[pipeline_id, fold_id] = scorer(pipeline, X_test, y_test)
                        bar()
            print(Fore.RESET, end='')
            self.__scores[dataset_path] = score
            save_file = self.__construct_file_name(dataset_path, False)
            np.save(save_file, score)
            SUCCESS(f"Result saved in {save_file}")

    def __analyze_statistics(self) -> None:
        # t_statistics = np.zeros((len(self.__pipelines), len(self.__pipelines)))
        # p_value = np.zeros((len(self.__pipelines), len(self.__pipelines)))
        WARNING("\n\n[2/2] Analyzing statistics...")
        for dataset in self.__scores.keys():
            self.__t_statistics[dataset] = np.zeros((len(self.__pipelines), len(self.__pipelines)))
            self.__p_values[dataset] = np.zeros((len(self.__pipelines), len(self.__pipelines)))
            for i in range(len(self.__pipelines)):
                for j in range(len(self.__pipelines)):
                    self.__t_statistics[dataset][i, j], self.__p_values[dataset][i, j] = ttest_rel(
                        self.__scores[dataset][i], self.__scores[dataset][j])
                    self.__t_statistics[dataset] = np.nan_to_num(self.__t_statistics[dataset])
                    self.__p_values[dataset] = np.nan_to_num(self.__p_values[dataset], nan=1.0)

            headers = self.__pipelines_credentials()
            column_names = np.array(list(map(lambda name: [name], headers)))
            t_statistic_table = np.concatenate((column_names, self.__t_statistics[dataset]), axis=1)
            t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
            p_value_table = np.concatenate((column_names, self.__p_values[dataset]), axis=1)
            p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
            INFO(f"{dataset}: t-statistic:\n{t_statistic_table}")
            SUCCESS(f"\n\np-value:\n{p_value_table}\n\n")

            advantage = np.zeros((len(pipelines), len(pipelines)))
            advantage[self.__t_statistics[dataset] > 0] = 1
            advantage_table = tabulate(np.concatenate(
                (column_names, advantage), axis=1)
                , headers)
            IMPORTANT(f"{dataset}: Advantage:\n{advantage_table}\n\n")

            significance = np.zeros((len(pipelines), len(pipelines)))
            significance[self.__p_values[dataset] <= self.__alpha] = 1
            significance_table = tabulate(np.concatenate(
                (column_names, significance), axis=1
            ), headers)
            INFO(f"{dataset}: Statistical significance (alpha = {self.__alpha}):\n{significance_table}\n\n")

            statistically_better = significance * advantage
            statistically_better_table = tabulate(np.concatenate(
                (column_names, statistically_better), axis=1
            ), headers)
            SUCCESS(f"{dataset}: Statistically significantly better:\n{statistically_better_table}")

    def __pipeline_credentials(self, pipeline: Pipeline) -> Tuple[str, List[str]]:
        steps = pipeline.get_params()['steps']
        classifier_name = steps[-1][0]  # classifier is always last tuple. Each tuple looks like: (name, model)
        preprocessing_techniques = [step[0] for step in steps[:-1]]

        return classifier_name, preprocessing_techniques

    def __pipelines_credentials(self) -> List[str]:
        pipelines_credentials = []
        for pipeline in self.__pipelines:
            classifier_name, preprocessings = self.__pipeline_credentials(pipeline)
            pipelines_credentials.append(f"{classifier_name} {', '.join(preprocessings)}")

        return pipelines_credentials

    def __construct_file_name(self, dataset_path: str, with_extension=True) -> str:
        dataset_name = self.__get_dataset_name(dataset_path)
        file_name = dataset_name + f"_{self.__started_at}"

        if with_extension:
            file_name += ".txt"

        return file_name

    def __get_dataset_name(self, dataset_path: str) -> str:
        path = Path(dataset_path)
        suffix = path.suffix
        dataset_name = path.name.removesuffix(suffix)

        return dataset_name

    def __show_config(self) -> None:
        self.__started_at = datetime.now().strftime(self.__dt_format)
        # files = [self.__construct_file_name(dataset_path) for dataset_path in self.__datasets]
        # files_info = f"Results will be saved in: {', '.join(files)}"
        pipelines_info = [["Classifier"], ["Preprocessing technique(s)"]]
        for pipeline in self.__pipelines:
            classifier, preprocessings = self.__pipeline_credentials(pipeline)
            pipelines_info[0].append(classifier)
            pipelines_info[1].append(', '.join(preprocessings))
        pipelines_info = tabulate(pipelines_info, tablefmt='fancy_grid')

        ##########
        INFO(self.__test_type)
        # INFO(files_info)
        WARNING(f"Provided classificators and techniques:\n{pipelines_info}")
        IMPORTANT("Caution! These tests might take a while! Be patient")

    def run(self) -> None:
        self.__show_config()
        self.__calculate_statistics()
        self.__analyze_statistics()


# def conduct_test(n_splits: int, n_repeats: int, file_name: str):
#     repeated_stratified_k_fold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
#
#     scores = np.zeros((len(pipelines), len(datasets), n_splits * n_repeats))
#
#     for dataset_id, dataset in enumerate(datasets):
#         measurement_start = time()
#         print(f"{Fore.YELLOW} Testing: {dataset}")
#         dataset_name = Path(dataset).name
#         contains_categorical_col = True if dataset_id == 0 else False
#         X, y = preprocess(dataset, 0 if contains_categorical_col else None)
#         for fold_id, (train, test) in enumerate(repeated_stratified_k_fold.split(X, y)):
#             X_train, X_test = X[train], X[test]
#             y_train, y_test = y[train], y[test]
#             X_train, sc = standard_scaler(X_train, contains_categorical_col)
#             if contains_categorical_col:
#                 X_test = sc.transform(X=X_test[:, 2:])
#             else:
#                 X_test = sc.transform(X=X_test)
#
#             for pipeline_id, pipeline in enumerate(pipelines):
#                 pipeline = clone(pipeline)
#                 pipeline.fit(X_train, y_train)
#                 scores[pipeline_id, dataset_id, fold_id] = scorer(pipeline, X_test, y_test)
#         measurement_stop = time()
#         execution_time = measurement_stop - measurement_start
#         print(f"{Fore.GREEN} {dataset_name} took: {execution_time} seconds")
#
#     np.save(file_name, scores)
#     print(f"{Fore.RESET}")  # reset colors
#
#
# def analyze_results(result_file_name: str) -> None:
#     scores = np.load(f"{result_file_name}.npy")
#     print(f"Scores: {scores.shape}")
#
#
# def perform_statistics(scores: np.array, alpha: float) -> str:
#     t_statistics = np.zeros((len(pipelines), len(pipelines)))
#     p_value = np.zeros((len(pipelines), len(pipelines)))
#     for i in range(len(pipelines)):
#         for j in range(len(pipelines)):
#             for k in range(len(datasets)):
#                 t_statistics[i, j], p_value[i, j] = ttest_rel(scores[i][k], scores[j][k])
#
#     t_statistics = np.nan_to_num(t_statistics)
#     p_value = np.nan_to_num(p_value, nan=1.0)
#
#     headers = models_description
#     column_names = np.array(list(map(lambda name: [name], headers)))
#     t_statistic_table = np.concatenate((column_names, t_statistics), axis=1)
#     t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
#     p_value_table = np.concatenate((column_names, p_value), axis=1)
#     p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
#     print(f"t-statistic:\n{t_statistic_table}\n\np-value:\n{p_value_table}\n\n")
#
#     advantage = np.zeros((len(pipelines), len(pipelines)))
#     advantage[t_statistics > 0] = 1
#     advantage_table = tabulate(np.concatenate(
#         (column_names, advantage), axis=1)
#         , headers)
#     print(f"Advantage:\n{advantage_table}\n\n")
#
#     significance = np.zeros((len(pipelines), len(pipelines)))
#     significance[p_value <= alpha] = 1
#     significance_table = tabulate(np.concatenate(
#         (column_names, significance), axis=1
#     ), headers)
#     print(f"Statistical significance (alpha = {alpha}):\n{significance_table}\n\n")
#
#     statistically_better = significance * advantage
#     statistically_better_table = tabulate(np.concatenate(
#         (column_names, statistically_better), axis=1
#     ), headers)
#     print(f"Statistically significantly better:\n{statistically_better_table}")
#
#
if __name__ == "__main__":
    # conduct_test(2, 5, "results")
    # analyze_results("results")
    # pipeline = [Pipeline(steps=[('Threaded BaggFold', None), ('SMA Oversampling', None), ('TBaggFold', ThreadedBaggFold(lambda: SVC()))])]
    test = StatisticalTest(datasets_paths(), pipelines, .05, 2, 5)
    test.run()
