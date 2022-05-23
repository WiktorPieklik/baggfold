import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler #, OneHotEncoder


def load_dataset(path):
    # load the dataset as a numpy array
    dataframe = pd.read_csv(path, na_values='?')
    # drop unnecessary columns
    #dataframe.drop(['Time'], axis=1)
    # change column names to numbers
    dataframe.columns = range(dataframe.shape[1])
    # drop rows with missing data
    dataframe = dataframe.dropna()
    # split into inputs and outputs
        
    # encode M,F in lung dataset
    if "cancer" in path:
        cleanup_nums = {0:     {"M": 0, "F": 1}}
        dataframe = dataframe.replace(cleanup_nums)

    last_id = len(dataframe.columns) - 1 # label column id
    X, y = dataframe.drop(last_id, axis=1), dataframe[last_id]
    # select categorical and numerical features
    #cat_ids = X.select_dtypes(include=['object', 'bool', 'string']).columns
    #num_ids = X.select_dtypes(include=['int64', 'float64']).columns

    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    # scale data
    X = StandardScaler().fit_transform(X.values)
    return X, y #, dataframe, cat_ids, num_ids