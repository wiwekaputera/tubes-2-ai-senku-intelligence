
import numpy as np

class DecisionTreeScratch:
    """
    A Decision Tree Classifier implemented from scratch using NumPy.
    Supports ID3 (Information Gain) or CART (Gini Impurity) algorithms.
    """
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        """
        Builds the decision tree classifier from the training set (X, y).
        """
        pass

    def predict(self, X):
        """
        Predict class for X.
        """
        pass