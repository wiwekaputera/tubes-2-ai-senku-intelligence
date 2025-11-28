import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    """
    Logistic Regression using Stochastic Gradient Ascent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        X_b = np.insert(X, 0, 1, axis=1)
        n_examples, n_features = X_b.shape
        
        self.b = np.zeros(n_features)

        for t in range(self.n_iterations):
            for i in range(n_examples):
                linear_combination = np.dot(X_b[i], self.b)
                
                p_i = self.sigmoid(linear_combination)
                error = y[i] - p_i

                # b_j = b_j + eta * (y_i - p_i) * x_ij
                self.b = self.b + self.learning_rate * error * X_b[i]



    def predict_proba(self, X):
        """
        Calculates the probability P(y=1|x,b) for each example.
        """
        X_b = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(np.dot(X_b, self.b))
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

class OneVsAll:
    """
    One-vs-All (OvA) for multi-class classification.
    """
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
        self.models = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []

        for k in self.classes:
            # Transformasi label target menjadi biner (OvA)
            y_k = np.where(y == k, 1, 0)
            
            model = self.model_class(**self.kwargs)
            # Latih model
            model.fit(X, y_k)
            self.models.append(model)
        return self


    def predict(self, X):
        all_probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_probas.append(proba)

        confusion_matrix = np.column_stack(all_probas)

        best_class_indices = np.argmax(confusion_matrix, axis=1)

        # Konversi indeks ke label asli
        predictions = self.classes[best_class_indices]
        
        return predictions



