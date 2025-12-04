import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

class SVMNode:
    """
    Binary SVM with mini batch training and early stopping
    """
    def __init__(self, learning_rate=0.001, reg_strength=0.01, max_epochs=1000,
                 batch_size=32, lr_decay=True, early_stopping=True, patience=50, tol=1e-5):
        self.lr_init = learning_rate
        self.lr = learning_rate
        self.reg = reg_strength 
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        
        self.weights = None
        self.bias = None
        
        # Training history for visualization (Ini buat bonus)
        self.loss_history = []
        self.accuracy_history = []

    def compute_margin(self, X):
        return np.dot(X, self.weights) + self.bias
    
    # Hitung dengan formula max(0, 1 - y*(w.x + b)) + reg*||w||^2
    def _hinge_loss(self, X, y_scaled):
        margins = y_scaled * self.compute_margin(X)
        hinge = np.maximum(0, 1 - margins)
        reg_term = 0.5 * self.reg * np.dot(self.weights, self.weights)
        return np.mean(hinge) + reg_term
    
    # Calculate accuracy
    def _accuracy(self, X, y_scaled):
        predictions = np.sign(self.compute_margin(X))
        return np.mean(predictions == y_scaled)

    def train_node(self, X_train, y_train):
        num_samples, num_features = X_train.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        y_scaled = np.where(y_train <= 0, -1, 1)
        
        # Reset history
        self.loss_history = []
        self.accuracy_history = []
        
        # Early stopping tracker
        best_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_bias = None
        
        # Batch count
        n_batches = max(1, num_samples // self.batch_size)

        for epoch in range(self.epochs):
            # LR Decay: lr = lr_init / (1 + decay_rate * epoch)
            if self.lr_decay:
                self.lr = self.lr_init / (1 + 0.01 * epoch)
            
            # Shuffle every epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_scaled[indices]
            
            # Mini batch training
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, num_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                batch_size_actual = len(y_batch)
                
                # Calculate marginnya
                margins = np.dot(X_batch, self.weights) + self.bias
                
                # Find misclassified samples (margin < 1)
                misclassified = (y_batch * margins) < 1
                
                # Compute gradients
                grad_w = self.reg * self.weights
                grad_b = 0
                
                if np.any(misclassified):
                    X_mis = X_batch[misclassified]
                    y_mis = y_batch[misclassified]
                    
                    grad_w -= np.dot(y_mis, X_mis) / batch_size_actual
                    grad_b -= np.sum(y_mis) / batch_size_actual
                
                # Update weights and bias
                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b
            
            current_loss = self._hinge_loss(X_train, y_scaled)
            current_acc = self._accuracy(X_train, y_scaled)
            
            self.loss_history.append(current_loss)
            self.accuracy_history.append(current_acc)
            
            # Early Stopping
            if self.early_stopping:
                if current_loss < best_loss - self.tol:
                    best_loss = current_loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    self.weights = best_weights
                    self.bias = best_bias
                    break
        
        return self

class SVMScratch:
    """
    Multiclass SVM using One-vs-All strategy
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000,
                 batch_size=32, lr_decay=True, early_stopping=True, patience=50):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.sub_classifiers = []
        self.known_classes = []
        
        # Store training history all classes(buat bonus)
        self.training_history = {}

    # One-vs-All
    def fit(self, features, targets): 
        self.sub_classifiers = []
        self.known_classes = np.unique(targets)
        self.training_history = {}
        
        n_classes = len(self.known_classes)
        print(f"SVM untuk {n_classes} kelas: {self.known_classes}")
        
        for cls in self.known_classes:
            print(f"  Training classifier for class {cls}...", end=" ")
            binary_labels = np.where(targets == cls, 1, -1)
            
            node = SVMNode(
                learning_rate=self.learning_rate,
                reg_strength=self.lambda_param,
                max_epochs=self.n_iters,
                batch_size=self.batch_size,
                lr_decay=self.lr_decay,
                early_stopping=self.early_stopping,
                patience=self.patience
            )

            node.train_node(features, binary_labels)
            
            # Store training history
            self.training_history[cls] = {
                'loss': node.loss_history,
                'accuracy': node.accuracy_history
            }
            
            epochs_trained = len(node.loss_history)
            final_acc = node.accuracy_history[-1] if node.accuracy_history else 0
            print(f"Done! (Epochs: {epochs_trained}, Acc: {final_acc:.4f})")

            self.sub_classifiers.append(node)
        
        return self

    def predict(self, features):
        if not self.sub_classifiers:
            raise Exception("Error: Model belum dilatih!")

        num_samples = features.shape[0]
        num_classes = len(self.known_classes)
        
        scores_matrix = np.zeros((num_samples, num_classes))

        for idx, node in enumerate(self.sub_classifiers):
            scores_matrix[:, idx] = node.compute_margin(features)

        winning_indices = np.argmax(scores_matrix, axis=1)
        return self.known_classes[winning_indices]
    
    # Bonus: Visualisasi proses trainingnya
    def plot_training_history(self, save_path=None, show=True):
        n_classes = len(self.known_classes)
        fig, axes = plt.subplots(2, n_classes, figsize=(5*n_classes, 8))
        
        class_names = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        
        for idx, cls in enumerate(self.known_classes):
            history = self.training_history[cls]
            epochs = range(1, len(history['loss']) + 1)
            
            # Loss plot
            ax_loss = axes[0, idx] if n_classes > 1 else axes[0]
            ax_loss.plot(epochs, history['loss'], 'b-', linewidth=1.5)
            ax_loss.set_title(f'Class: {class_names.get(cls, cls)} (OvA)\nHinge Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax_acc = axes[1, idx] if n_classes > 1 else axes[1]
            ax_acc.plot(epochs, history['accuracy'], 'g-', linewidth=1.5)
            ax_acc.set_title(f'Binary Accuracy')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_ylim([0, 1.05])
            ax_acc.grid(True, alpha=0.3)
        
        plt.suptitle('SVM Training Progress (One-vs-All)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    # Get summary stats
    def get_training_summary(self):
        summary = {}
        for cls in self.known_classes:
            history = self.training_history[cls]
            summary[cls] = {
                'epochs_trained': len(history['loss']),
                'final_loss': history['loss'][-1] if history['loss'] else None,
                'final_accuracy': history['accuracy'][-1] if history['accuracy'] else None,
                'best_accuracy': max(history['accuracy']) if history['accuracy'] else None,
            }
        return summary

    # Save model dengan pickle
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        # Load model from file
        with open(filepath, 'rb') as f:
            return pickle.load(f)