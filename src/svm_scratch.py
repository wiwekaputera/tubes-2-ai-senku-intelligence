import cupy as cp
import numpy as np
import json

class SVMScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classifiers = []
        self.classes = None

    def fit(self, X, y):
        # 1. Convert Inputs to GPU (CuPy)
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        
        self.classes = cp.unique(y_gpu)
        self.classifiers = []
        
        print(f"🚀 Training SVM on GPU for classes: {self.classes}")
        
        # Train One-vs-All
        for cls in self.classes:
            # Create Binary Targets (-1 vs 1)
            y_binary = cp.where(y_gpu == cls, 1, -1)
            
            node = SVMNode(self.lr, self.lambda_param, self.n_iters)
            node.train(X_gpu, y_binary)
            self.classifiers.append(node)

    def predict(self, X):
        # 1. Convert Input to GPU
        X_gpu = cp.asarray(X)
        
        n_samples = X_gpu.shape[0]
        n_classes = len(self.classes)
        scores = cp.zeros((n_samples, n_classes))
        
        # 2. Get scores from all classifiers
        for idx, node in enumerate(self.classifiers):
            scores[:, idx] = node.predict_score(X_gpu)
            
        # 3. Find max score
        winning_indices = cp.argmax(scores, axis=1)
        
        # 4. Convert Result back to CPU (NumPy) for compatibility
        return cp.asnumpy(self.classes[winning_indices])

class SVMNode:
    def __init__(self, lr, reg, epochs):
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = cp.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            # --- FULLY VECTORIZED UPDATE (No Loops) ---
            
            # Calculate margins
            margins = y * (cp.dot(X, self.weights) + self.bias)
            
            # Identify misclassified points
            misclassified_mask = margins < 1
            
            # Gradient of Regularization term
            dw = 2 * self.reg * self.weights
            db = 0
            
            # Gradient of Hinge Loss (only for misclassified)
            if cp.any(misclassified_mask):
                # sum(-y_i * x_i)
                # We use matrix multiplication: X[mask].T @ y[mask]
                X_mis = X[misclassified_mask]
                y_mis = y[misclassified_mask]
                
                dw -= cp.dot(X_mis.T, y_mis)
                db -= cp.sum(y_mis)
                
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_score(self, X):
        return cp.dot(X, self.weights) + self.bias