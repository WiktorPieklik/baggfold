from __future__ import annotations

from evaluate import datasets_paths, scorer, standard_scaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
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
        dataset_name = path.name[:-4]

        return dataset_name

    def __show_config(self) -> None:
        self.__started_at = datetime.now().strftime(self.__dt_format)
        pipelines_info = [["Classifier"], ["Preprocessing technique(s)"]]
        for pipeline in self.__pipelines:
            classifier, preprocessings = self.__pipeline_credentials(pipeline)
            pipelines_info[0].append(classifier)
            pipelines_info[1].append(', '.join(preprocessings))
        pipelines_info = tabulate(pipelines_info, tablefmt='fancy_grid')

        ##########
        INFO(self.__test_type)
        WARNING(f"Provided classificators and techniques:\n{pipelines_info}")
        IMPORTANT("Caution! These tests might take a while! Be patient")

    def run(self) -> None:
        self.__show_config()
        self.__calculate_statistics()
        self.__analyze_statistics()


if __name__ == "__main__":
    classifiers_types = {
        # 'C45': C45(), # very poor implementation. Way to slow to conduct tests!!!
        'CART': DecisionTreeClassifier(),
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
    # reference methods
    for classifier in classifiers_types.items():
        for technique in techniques.items():
            steps = [(technique[0], technique[1]), (classifier[0], classifier[1])]
            pipelines.append(Pipeline(steps=steps))
    # BaggFold SMA Oversampling
    pipelines.append(Pipeline(steps=[('NONE', None), ('BaggFold', ThreadedBaggFold(lambda: SVC()))]))
    test = StatisticalTest(datasets_paths(), pipelines, .05, 2, 5)
    test.run()
