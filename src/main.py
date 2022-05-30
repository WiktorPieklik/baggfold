from evaluate import datasets_paths, evaluate_reference_models, evaluate_model
from preprocessing import preprocess
from baggfold import BaggFold
from sklearn.svm import SVC


if __name__ == "__main__":
    lung_cancer, diabetes, credit_cards = datasets_paths()
    X, y = preprocess(diabetes)
    evaluate_reference_models(X, y)

    # BaggFold SMA Oversampling
    steps = [('model', BaggFold(lambda: SVC()))]
    description = ('BaggFold', 'SMA Oversampling')
    evaluate_model(steps=steps, X=X, y=y, model_description=description)

