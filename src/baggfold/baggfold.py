import numpy as np

from math import ceil
from random import choice
from typing import Tuple, Callable
from abc import ABC, abstractmethod
from threading import Thread
from imblearn.over_sampling import SMOTE


def get_synthetic_samples(smote_x: np.array, original_x: np.array) -> np.array:
    ncols = original_x.shape[1]
    synthetic_samples = []
    dtype = {'names': [f"f{i}" for i in range(ncols)],
             'formats': ncols * [smote_x.dtype]}
    common_x = np.intersect1d(smote_x.astype(dtype), original_x.astype(dtype)).view(smote_x.dtype).reshape(-1, ncols)
    for row in smote_x:
        if row.tolist() not in common_x.tolist():
            synthetic_samples.append(row.tolist())

    return np.array(synthetic_samples).reshape(-1, ncols)


def choices(sequence: np.array, k: int) -> np.array:
    new_array = np.array([])
    for _ in range(k):
        new_array = np.concatenate((new_array, choice(sequence)))

    return np.reshape(new_array, (-1, sequence.shape[1]))


def predict(classifiers: list, X: np.array, aggregate: bool = True) -> np.array:
    classifications = np.array([])
    for classifier in classifiers:
        classifications = np.concatenate((classifications, classifier.predict(X)))
    classifications = np.reshape(classifications, (len(classifiers), -1))

    if aggregate:
        voting = classifications.mean(axis=0)
        return np.where(voting >= .5, 1, 0)
    else:
        return classifications


class BaseBaggFold(ABC):
    def __init__(self, base_classifier_fn: Callable[[], object]):
        self.x_majority = None
        self.y_majority = None
        self.x_minority = None
        self.y_minority = None
        self._needed_classificators_count = 0
        self._classifiers = []
        self.__init_classifier = base_classifier_fn

    def fit(self, X: np.array, y: np.array) -> None:
        self._detect_samples_distribution(X, y)
        self._set_classificators_count()
        self._prepare_dataset()
        self._instantiate_fit_classificators()

    @abstractmethod
    def predict(self, X) -> np.array:
        raise NotImplemented

    @property
    def minority_count(self) -> int:
        return 0 if self.y_minority is None else self.y_minority.size

    @property
    def majority_count(self) -> int:
        return 0 if self.y_majority is None else self.y_majority.size

    def _detect_samples_distribution(self, X: np.array, y: np.array) -> None:
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

    def _set_classificators_count(self) -> None:
        self._needed_classificators_count = ceil(self.majority_count / self.minority_count)

    def _oversample_majority(self, x_majority: np.array, y_majority: np.array) -> Tuple[np.array, np.array]:
        smote = SMOTE(sampling_strategy=1.0)
        x = np.concatenate((x_majority, self.x_minority), axis=0)
        y = np.concatenate((y_majority, self.y_minority), axis=0)

        return smote.fit_resample(x, y)

    def _prepare_dataset(self) -> None:
        oversampling_count = self.minority_count * ceil(self.majority_count / self.minority_count) - self.majority_count
        if oversampling_count > 5:
            smote = SMOTE(sampling_strategy=1.0)
            x_majority = choices(self.x_minority,
                                 k=oversampling_count * 2)  # majority in terms of smote. In fact, it's the
            # minority class
            y_majority = np.array([self.y_minority[0]] * (2 * oversampling_count))
            x_minority = self.x_majority[:oversampling_count, :]
            y_minority = self.y_majority[:oversampling_count]
            x = np.concatenate((x_majority, x_minority), axis=0)
            y = np.concatenate((y_majority, y_minority), axis=0)

            smote_x, smote_y = smote.fit_resample(x, y)
            synthetic_x = get_synthetic_samples(smote_x, x)
            self.x_majority = np.concatenate((self.x_majority, synthetic_x), axis=0)
            self.y_majority = np.append(self.y_majority, [self.y_majority[0]] * synthetic_x.shape[0])

    def _instantiate_fit_classificators(self) -> None:
        start_index = 0
        minority_count = self.y_minority.size
        self._classifiers = []
        for _ in range(self._needed_classificators_count):
            classifier = self.__init_classifier()
            end_index = start_index + minority_count
            x_majority = self.x_majority[start_index:end_index, :]
            y_majority = self.y_majority[start_index:end_index]
            x = np.concatenate((x_majority, self.x_minority), axis=0)
            y = np.concatenate((y_majority, self.y_minority), axis=0)
            classifier.fit(x, y)
            self._classifiers.append(classifier)
            start_index += minority_count


class BaggFold(BaseBaggFold):
    def predict(self, X) -> np.array:
        return predict(self._classifiers, X)


class BaggFoldThread(Thread):
    def __init__(self, classifiers: list, X: np.array):
        super().__init__()
        self.__classifiers = classifiers
        self.__X = X
        self.predictions = None

    def run(self) -> None:
        self.predictions = predict(self.__classifiers, self.__X, False)


class ThreadedBaggFold(BaseBaggFold):
    def __init__(self, base_classifier_fn: Callable[[], object], max_threads: int = 30):
        super().__init__(base_classifier_fn)
        self.__threads = []
        self.__max_threads = max_threads

    def predict(self, X) -> np.array:
        preferred_threads_count = self.__max_threads if self._needed_classificators_count >= 30 else 5
        self.__prepare_threads(preferred_threads_count, X)
        classifications = np.array([])
        for thread in self.__threads:
            thread.start()
        for thread in self.__threads:
            thread.join()
            classifications = np.append(classifications, thread.predictions)

        classifications = np.reshape(classifications, (self._needed_classificators_count, -1))
        voting = classifications.mean(axis=0)

        return np.where(voting >= .5, 1, 0)

    def __prepare_threads(self, threads_count: int, X: np.array) -> int:
        classifiers_per_thread = ceil(self._needed_classificators_count / threads_count)
        start_index = 0
        self.__threads = []
        classifiers_left = self._needed_classificators_count
        for _ in range(threads_count):
            end_index = start_index + classifiers_per_thread
            classifiers = self._classifiers[start_index:end_index]
            thread = BaggFoldThread(classifiers, X)
            self.__threads.append(thread)
            start_index += classifiers_per_thread
            classifiers_left -= len(classifiers)
            if classifiers_left == 0:
                break

        return len(self.__threads)