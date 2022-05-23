from math import ceil
from typing import Tuple
import numpy as np
from src.c45 import C45
from imblearn.over_sampling import SMOTE


class BaggFold:
    def __init__(self):
        self.x_majority = None
        self.y_majority = None
        self.x_minority = None
        self.y_minority = None
        self.__needed_classificators_count = 0
        self.__classifiers = []

    def fit(self, X: np.array, y: np.array):
        self.__detect_samples_distributions(X, y)
        self.__set_classificators_count()
        self.__instantiate_fit_classificators()

    def predict(self, X) -> np.array:
        classifications = np.array([])
        for classifier in self.__classifiers:
            classifications = np.concatenate((classifications, classifier.predict(X)))
        classifications = np.reshape(classifications, (self.__needed_classificators_count, -1))

        voting = classifications.mean(axis=0)
        return np.where(voting >= .5, 1, 0)

    def __detect_samples_distributions(self, X: np.array, y: np.array) -> None:
        positive_samples_indices = np.where(y == 1)[0]
        negative_samples_indices = np.where(y == 0)[0]
        positives_count = len(positive_samples_indices)
        negatives_count = len(negative_samples_indices)

        if positives_count > negatives_count:
            self.x_majority = X[positive_samples_indices, :]
            self.y_majority = y[positive_samples_indices]
            self.x_minority = X[negative_samples_indices, :]
            self.y_minority = y[negative_samples_indices]
        else:
            self.x_majority = X[negative_samples_indices, :]
            self.y_majority = y[negative_samples_indices]
            self.x_minority = X[positive_samples_indices, :]
            self.y_minority = y[positive_samples_indices]

    def __set_classificators_count(self) -> None:
        self.__needed_classificators_count = ceil(self.y_majority.size / self.y_minority.size)

    def __oversample_majority(self, x_majority: np.array, y_majority: np.array) -> Tuple[np.array, np.array]:
        smote = SMOTE(sampling_strategy=1.0)
        x = np.concatenate((x_majority, self.x_minority), axis=0)
        y = np.concatenate((y_majority, self.y_minority), axis=0)

        return smote.fit_resample(x, y)

    def __instantiate_fit_classificators(self) -> None:
        start_index = 0
        minority_count = self.y_minority.size
        self.__classifiers = []
        for _ in range(self.__needed_classificators_count):
            classifier = C45()
            end_index = start_index + minority_count
            x_majority = self.x_majority[start_index:end_index, :]
            y_majority = self.y_majority[start_index:end_index]
            x = np.concatenate((x_majority, self.x_minority), axis=0)
            y = np.concatenate((y_majority, self.y_minority), axis=0)
            if y.size != 2 * minority_count:
                x, y = self.__oversample_majority(x_majority, y_majority)
            classifier.fit(x, y)
            self.__classifiers.append(classifier)
            start_index += minority_count
