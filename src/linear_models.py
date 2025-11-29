import numpy as np
import pickle


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 batch_size=32, lambda_reg=0.01, decay_rate=0.0,
                 class_weight=None, early_stopping=False, patience=50,
                 tol=1e-5):
        """
        Enhanced Logistic Regression with early stopping.
        
        Parameters:
        -----------
        learning_rate : float
            Initial learning rate (eta_0)
        n_iterations : int
            Maximum number of epochs
        batch_size : int
            Size of mini-batches. Use -1 for full-batch GD.
        lambda_reg : float
            L2 regularization strength
        decay_rate : float
            Learning rate decay. lr = lr_0 / (1 + decay_rate * epoch)
        class_weight : None, 'balanced', or dict
            Weights for each class to handle imbalance
        early_stopping : bool
            If True, stop when validation loss stops improving
        patience : int
            Number of epochs to wait before early stopping
        tol : float
            Minimum improvement threshold for early stopping
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.decay_rate = decay_rate
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []
        self._sample_weights = None
        self.best_weights = None
        self.best_bias = None

    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        z = np.clip(z, -500, 500)
        positive_mask = z >= 0
        result = np.zeros_like(z, dtype=np.float64)
        result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
        exp_z = np.exp(z[~positive_mask])
        result[~positive_mask] = exp_z / (1.0 + exp_z)
        return result

    def _compute_sample_weights(self, y):
        """Compute sample weights based on class weights."""
        if self.class_weight is None:
            return np.ones(len(y))
        elif self.class_weight == 'balanced':
            classes, counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            class_weights = {c: n_samples / (n_classes * count) 
                            for c, count in zip(classes, counts)}
            return np.array([class_weights[int(yi)] for yi in y])
        elif isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(int(yi), 1.0) for yi in y])
        return np.ones(len(y))

    def _compute_loss(self, X, y, sample_weights=None):
        """Compute Binary Cross-Entropy Loss with L2 regularization."""
        n = len(y)
        z = np.dot(X, self.weights) + self.bias
        p = self._sigmoid(z)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        if sample_weights is None:
            sample_weights = np.ones(n)
        
        bce = -np.sum(sample_weights * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(sample_weights)
        l2_term = (self.lambda_reg / 2) * np.sum(self.weights ** 2)
        return bce + l2_term
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        sample_weights = self._compute_sample_weights(y)
        
        # Xavier initialization
        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # For early stopping
        best_loss = np.inf
        patience_counter = 0
        self.best_weights = self.weights.copy()
        self.best_bias = self.bias
        
        batch_size = n_samples if self.batch_size == -1 else min(self.batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(self.n_iterations):
            current_lr = self.learning_rate / (1.0 + self.decay_rate * epoch)
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            w_shuffled = sample_weights[indices]
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                w_batch = w_shuffled[start:end]
                
                z = np.dot(X_batch, self.weights) + self.bias
                p = self._sigmoid(z)
                
                error = (p - y_batch) * w_batch
                total_weight = np.sum(w_batch)
                
                grad_w = np.dot(X_batch.T, error) / total_weight
                grad_w += self.lambda_reg * self.weights
                grad_b = np.sum(error) / total_weight
                
                self.weights -= current_lr * grad_w
                self.bias -= current_lr * grad_b

            # Early stopping check
            if epoch % 10 == 0 or epoch == self.n_iterations - 1:
                loss = self._compute_loss(X, y, sample_weights)
                self.loss_history.append((epoch, loss))
                
                if self.early_stopping:
                    if loss < best_loss - self.tol:
                        best_loss = loss
                        patience_counter = 0
                        self.best_weights = self.weights.copy()
                        self.best_bias = self.bias
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience // 10:
                        # Restore best weights
                        self.weights = self.best_weights
                        self.bias = self.best_bias
                        break

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def save(self, filepath):
        """Save model to file using pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class OneVsAll:
    def __init__(self, model_class, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs
        self.models = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []

        for k in self.classes:
            y_binary = np.where(y == k, 1, 0)
            
            model = self.model_class(**self.kwargs)
            model.fit(X, y_binary)
            self.models.append(model)
            
        return self

    def predict_proba(self, X):
        all_probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_probas.append(proba)
        
        return np.column_stack(all_probas)

    def predict(self, X):
        proba_matrix = self.predict_proba(X)
        best_class_indices = np.argmax(proba_matrix, axis=1)
        return self.classes[best_class_indices]
    
    def get_loss_history(self):
        histories = {}
        for idx, (cls, model) in enumerate(zip(self.classes, self.models)):
            if hasattr(model, 'loss_history'):
                histories[f"class_{cls}"] = model.loss_history
        return histories
    
    def save(self, filepath):
        """Save model to file using pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)



