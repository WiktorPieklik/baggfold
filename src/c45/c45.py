# Source code: https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier

from xml.dom import minidom
from xml.etree import ElementTree as ET

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .c45_utils import decision, grow_tree


class C45(BaseEstimator, ClassifierMixin):
    def __init__(self, attrNames=None):
        if attrNames is not None:
            attrNames = [''.join(i for i in x if i.isalnum()).replace(' ', '_') for x in attrNames]
        self.attrNames = attrNames
        self.X_ = None
        self.y_ = None
        self.tree_ = None
        self.resultType = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.resultType = type(y[0])
        if self.attrNames is None:
            self.attrNames = [f'attr{x}' for x in range(len(self.X_[0]))]

        assert (len(self.attrNames) == len(self.X_[0]))

        data = [[] for i in range(len(self.attrNames))]
        categories = []

        for i in range(len(self.X_)):
            categories.append(str(self.y_[i]))
            for j in range(len(self.attrNames)):
                data[j].append(self.X_[i][j])
        root = ET.Element('DecisionTree')
        grow_tree(data, categories, root, self.attrNames)
        self.tree_ = ET.tostring(root, encoding="unicode")
        return self

    def predict(self, X):
        check_is_fitted(self, ['tree_', 'resultType', 'attrNames'])
        X = check_array(X)
        dom = minidom.parseString(self.tree_)
        root = dom.childNodes[0]
        prediction = []
        for i in range(len(X)):
            answerlist = decision(root, X[i], self.attrNames, 1)
            answerlist = sorted(answerlist.items(), key=lambda x: x[1], reverse=True)
            answer = answerlist[0][0]
            prediction.append((self.resultType)(answer))
        return prediction

    def printTree(self):
        check_is_fitted(self, ['tree_'])
        dom = minidom.parseString(self.tree_)
        print(dom.toprettyxml(newl="\r\n"))
