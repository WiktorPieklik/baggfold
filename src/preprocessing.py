import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def preprocess(dataset_path: str, categorical_data_column: Optional[int] = None) -> Tuple[np.array, np.array]:
    # retrieving data
    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # instantiating helpers
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [categorical_data_column])],
                           remainder='passthrough')
    le = LabelEncoder()
    sc = StandardScaler()
    if categorical_data_column is not None:
        X[:, categorical_data_column + 1:-1] = imputer.fit_transform(X=X[:, categorical_data_column + 1:-1])
        X = np.array(ct.fit_transform(X))
    else:
        X[:, :-1] = imputer.fit_transform(X=X[:, :-1])
    y = le.fit_transform(y=y)

    if categorical_data_column is not None:
        X[:, 2:] = sc.fit_transform(X=X[:, 2:])
    else:
        X = sc.fit_transform(X=X)

    return X, y
