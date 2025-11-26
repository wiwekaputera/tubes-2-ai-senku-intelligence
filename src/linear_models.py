import cupy as cp
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + cp.exp(-z))

    def fit(self, X, y):
        # 1. Convert to GPU
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        
        n_samples, n_features = X_gpu.shape
        self.weights = cp.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            linear_model = cp.dot(X_gpu, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Vectorized Gradients
            dw = (1 / n_samples) * cp.dot(X_gpu.T, (y_pred - y_gpu))
            db = (1 / n_samples) * cp.sum(y_pred - y_gpu)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        # Handle input that might be CPU or GPU
        X_gpu = cp.asarray(X)
        linear_model = cp.dot(X_gpu, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        preds = (probs > threshold).astype(int)
        # Return to CPU
        return cp.asnumpy(preds)

class OneVsAllClassifier:
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
        self.models = []
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y) # Keep classes metadata on CPU
        self.models = []
        
        print(f"🚀 Training LogReg (OvA) on GPU for classes: {self.classes}")
        
        for c in self.classes:
            # Create binary target (CPU numpy array first)
            y_binary = np.where(y == c, 1, 0)
            
            # Model.fit will handle the GPU conversion internally
            model = self.model_class(**self.kwargs)
            model.fit(X, y_binary)
            self.models.append(model)
            
    def predict(self, X):
        if not self.models:
            raise Exception("Model not trained")
            
        # We need to collect probabilities from all models
        # Convert X to GPU once for efficiency
        X_gpu = cp.asarray(X)
        n_samples = X_gpu.shape[0]
        n_classes = len(self.classes)
        
        probas = cp.zeros((n_samples, n_classes))
        
        for idx, model in enumerate(self.models):
            # model.predict_proba returns CuPy array
            probas[:, idx] = model.predict_proba(X_gpu)
            
        winning_indices = cp.argmax(probas, axis=1)
        
        # Convert indices back to CPU to map to classes
        return self.classes[cp.asnumpy(winning_indices)]