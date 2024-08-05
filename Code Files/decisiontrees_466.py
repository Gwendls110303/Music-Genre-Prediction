'''Decision Trees'''
# https://scikit-learn.org/stable/modules/tree.html
from sklearn import tree

def main(X_train, X_test, y_train):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    return clf