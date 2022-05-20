from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict #, cross_val_score 
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hmeasure import h_score
from xgboost import XGBClassifier
from c45 import C45


def evaluate_models(X, y):
    # define models
    models = []
    models.append(('C45', C45()))
    models.append(('SVM', SVC())) # gamma='scale'
    models.append(('XGB', XGBClassifier()))

    # define oversampling techniques
    oversampling = []
    oversampling.append(('NONE', None))
    oversampling.append(('SMOTE', SMOTE()))   
    oversampling.append(('rnd_over', RandomOverSampler())) # sampling_strategy=0.1

    # define undersampling techniques
    undersampling = []
    oversampling.append(('rnd_under', RandomUnderSampler())) # sampling_strategy=0.5

    for model in models:
    
        for over in oversampling:
            #define pipeline
            steps = [('over', over[1]), ('model', model[1])]
            pipeline = Pipeline(steps=steps)

            # 5 times 2-fold cross-validation
            cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1) # n_repeats=5

            # roc-auc
            #scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, verbose=2)#, n_jobs=-1)
            #score = mean(scores)

            # h-measure
            predictions = cross_val_predict(pipeline, X, y, cv=cv, verbose=0)#, n_jobs=-1)
            n1, n0 = y.sum(), y.shape[0]-y.sum()
            score = h_score(y, predictions, severity_ratio=(n1/n0))
            
            print(f"model: {model[0]}-{over[0]} \t h-score: {score}")


        for under in undersampling:
            #define pipeline
            steps = [('under', under[1]), ('model', model[1])]
            pipeline = Pipeline(steps=steps)

            # 5 times 2-fold cross-validation
            cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=1) # n_repeats=5

            # roc-auc
            #scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, verbose=2)#, n_jobs=-1)
            #score = mean(scores)

            # h-measure
            predictions = cross_val_predict(pipeline, X, y, cv=cv, verbose=0)#, n_jobs=-1)
            n1, n0 = y.sum(), y.shape[0]-y.sum()
            score = h_score(y, predictions, severity_ratio=(n1/n0))
            
            print(f"model: {model[0]}-{under[0]} \t h-score: {score}")
        
        