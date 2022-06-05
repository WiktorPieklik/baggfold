from evaluate import datasets_paths, evaluate_reference_models, evaluate_model
from preprocessing import preprocess
from baggfold import ThreadedBaggFold
from sklearn.svm import SVC
from promting import WARNING
from pathlib import Path


if __name__ == "__main__":
    steps = [('model', ThreadedBaggFold(lambda: SVC()))]
    description = ('BaggFold', 'SMA Oversampling')
    for dataset_path in datasets_paths():
        dataset_name = Path(dataset_path).name[:-4]
        contains_categorical_column = False
        categorical_column = None
        if "lung_cancer" in dataset_path:
            contains_categorical_column = True
            categorical_column = 0
        X, y = preprocess(dataset_path, categorical_column)
        WARNING(f"Evaluating reference methods on {dataset_name}")
        evaluate_reference_models(X, y, contains_categorical_column, f"{dataset_name}.csv")
        WARNING(f"Evaluating BaggFold SMA Oversampling on {dataset_name}")
        model_name, score = evaluate_model(steps, X, y, description, contains_categorical_column)
        with open(f"{dataset_name}_baggfold.csv", "w") as file:
            file.write("Model;Score\n")
            file.write(f"{model_name};{score}\n")

