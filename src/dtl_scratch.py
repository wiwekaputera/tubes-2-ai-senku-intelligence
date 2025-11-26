import numpy as np
import pickle

class DecisionTreeScratch:
    """DTL CART implementation"""
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1): 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _gini(self, y):
        """Calculate Gini impurity (Optimized)"""
        n = len(y)
        if n == 0:
            return 0
        
        # OPTIMIZATION: Use bincount instead of unique (much faster for ints)
        # Ensure y is int type. preprocessing.py should guarantee this.
        counts = np.bincount(y.astype(int))
        
        # Remove zeros (for classes that aren't present in this node)
        counts = counts[counts > 0]
        
        probs = counts / n
        return 1 - np.sum(probs ** 2)

    def _is_categorical(self, values):
        """Checking if it's categorical (unique values <= 10)"""
        unique_vals = np.unique(values[~np.isnan(values)])
        return len(unique_vals) <= 10   

    def _split(self, X, y, feature_idx, threshold, is_categorical):
        """Split dataset from feature type"""
        if is_categorical:
            left_mask = X[:, feature_idx] == threshold
        else:
            left_mask = X[:, feature_idx] <= threshold
        
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):  
        """Finding best split"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_is_categorical = False
        
        parent_impurity = self._gini(y)
        n_samples = len(y)
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            is_categorical = self._is_categorical(feature_values)
            
            valid_mask = ~np.isnan(feature_values)
            thresholds = np.unique(feature_values[valid_mask])
            
            if not is_categorical and len(thresholds) > 20: # Take sample threshold if too many
                thresholds = np.percentile(feature_values[valid_mask], np.linspace(0, 100, 20))
            
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold, is_categorical)
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                n_left, n_right = len(y_left), len(y_right)
                left_impurity = self._gini(y_left)
                right_impurity = self._gini(y_right)
                
                weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain: # Update when found better
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_is_categorical = is_categorical
        
        return best_feature, best_threshold, best_is_categorical

    def _build_tree(self, X, y, depth=0):
        """Building tree recursively"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        majority_class = np.bincount(y.astype(int)).argmax()
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split or \
           n_samples < 2 * self.min_samples_leaf:
            return {'leaf': True, 'value': majority_class}
        
        feature_idx, threshold, is_categorical = self._best_split(X, y)
        
        if feature_idx is None:
            return {'leaf': True, 'value': majority_class}
        
        X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold, is_categorical)
        
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'is_categorical': is_categorical,
            'majority_class': majority_class,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """Fitting model to training data"""
        self.tree = self._build_tree(X, y)
        return self

    def _predict_single(self, x, node):
        """Predict for single sample"""
        if node['leaf']:
            return node['value']
        
        feature_val = x[node['feature']]

        if np.isnan(feature_val):
            return node['majority_class'] 
        
        # Determining the direction of split
        if node['is_categorical']:
            go_left = feature_val == node['threshold']
        else:
            go_left = feature_val <= node['threshold']
        
        if go_left:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        """Predict target values"""
        return np.array([self._predict_single(x, self.tree) for x in X])

    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)