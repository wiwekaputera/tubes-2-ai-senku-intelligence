import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 batch_size=32, lambda_reg=0.01, decay_rate=0.0):
        """
        Parameters:
        -----------
        learning_rate : float
            Initial learning rate (eta_0)
        n_iterations : int
            Number of epochs (full passes over the data)
        batch_size : int
            Size of mini-batches. Use -1 for full-batch GD.
        lambda_reg : float
            L2 regularization strength. Higher = more regularization.
        decay_rate : float
            Learning rate decay. lr = lr_0 / (1 + decay_rate * epoch)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.decay_rate = decay_rate
        self.weights = None
        self.bias = None
        self.loss_history = []  # For visualization/debugging

    def _sigmoid(self, z):
        """
        Numerically stable sigmoid function.
        Prevents overflow by clipping z and using conditional formula.
        """
        # Clip to prevent overflow in exp()
        z = np.clip(z, -500, 500)
        
        # Use stable formula: 
        # For z >= 0: 1 / (1 + exp(-z))
        # For z < 0:  exp(z) / (1 + exp(z))  [avoids large exp(-z)]
        positive_mask = z >= 0
        result = np.zeros_like(z, dtype=np.float64)
        
        # Stable for positive z
        result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
        
        # Stable for negative z
        exp_z = np.exp(z[~positive_mask])
        result[~positive_mask] = exp_z / (1.0 + exp_z)
        
        return result

    def _compute_loss(self, X, y):
        """
        Compute Binary Cross-Entropy Loss with L2 regularization.
        Loss = -1/n * Σ[y*log(p) + (1-y)*log(1-p)] + (λ/2n) * ||w||²
        """
        n = len(y)
        z = np.dot(X, self.weights) + self.bias
        p = self._sigmoid(z)
        
        # Clip probabilities to prevent log(0)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        
        # Binary Cross-Entropy
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        # L2 Regularization term (don't regularize bias)
        l2_term = (self.lambda_reg / (2 * n)) * np.sum(self.weights ** 2)
        
        return bce + l2_term
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights (Xavier/Glorot initialization for better convergence)
        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # Determine actual batch size
        batch_size = n_samples if self.batch_size == -1 else min(self.batch_size, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

        for epoch in range(self.n_iterations):
            # Learning rate decay: lr = lr_0 / (1 + decay * t)
            current_lr = self.learning_rate / (1.0 + self.decay_rate * epoch)
            
            # Shuffle data at the start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                batch_len = len(y_batch)
                
                # Forward pass
                z = np.dot(X_batch, self.weights) + self.bias
                p = self._sigmoid(z)
                
                # Compute gradients (vectorized)
                error = p - y_batch  # shape: (batch_size,)
                
                # Gradient of weights: (1/m) * X^T @ error + λ * w
                grad_w = (1.0 / batch_len) * np.dot(X_batch.T, error)
                grad_w += self.lambda_reg * self.weights  # L2 regularization
                
                # Gradient of bias: (1/m) * Σ(error) (no regularization on bias)
                grad_b = np.mean(error)
                
                self.weights -= current_lr * grad_w
                self.bias -= current_lr * grad_b

            if epoch % 100 == 0 or epoch == self.n_iterations - 1:
                loss = self._compute_loss(X, y)
                self.loss_history.append((epoch, loss))

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


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



