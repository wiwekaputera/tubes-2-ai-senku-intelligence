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

    def _split(self, X, y, feature_idx, threshold):
        """Split dataset (Vectorized and simplified)"""
        # Since we treat everything as numeric thresholds (even binary <= 0.5), 
        # we can unify this.
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y, feat_types):  
        """Finding best split with pre-computed feature types"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_impurity = self._gini(y)
        n_samples = len(y)
        n_features = X.shape[1]
        
        # Random feature subsampling (mtry) - optional but good for Random Forest later
        # features_to_check = range(n_features) 
        
        for feature_idx in range(n_features):
            # Skip if column is constant
            
            f_type = feat_types[feature_idx]
            
            # OPTIMIZATION: Handle Binary Features efficiently
            if f_type == 'binary':
                # Binary features (0/1) only need one split: <= 0.5
                # Left: 0, Right: 1
                thresholds = [0.5]
            else:
                # Continuous / Categorical with many values
                feature_values = X[:, feature_idx]
                valid_mask = ~np.isnan(feature_values)
                unique_vals = np.unique(feature_values[valid_mask])
                
                if len(unique_vals) < 2:
                    continue # Constant feature
                
                if len(unique_vals) > 20:
                    # Percentile approximation for continuous
                    thresholds = np.percentile(feature_values[valid_mask], np.linspace(5, 95, 20))
                else:
                    # Check all unique values (midpoints could be better, but exact values ok for now)
                    thresholds = unique_vals
            
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold)
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                n_left, n_right = len(y_left), len(y_right)
                left_impurity = self._gini(y_left)
                right_impurity = self._gini(y_right)
                
                weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _build_tree(self, X, y, feat_types, depth=0):
        """Building tree recursively"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Safe check for empty y
        if n_samples == 0:
             return {'leaf': True, 'value': None}

        majority_class = np.bincount(y.astype(int)).argmax()
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split or \
           n_samples < 2 * self.min_samples_leaf:
            return {'leaf': True, 'value': majority_class}
        
        feature_idx, threshold = self._best_split(X, y, feat_types)
        
        if feature_idx is None:
            return {'leaf': True, 'value': majority_class}
        
        X_left, y_left, X_right, y_right = self._split(X, y, feature_idx, threshold)
        
        left_subtree = self._build_tree(X_left, y_left, feat_types, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, feat_types, depth + 1)
        
        return {
            'leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'majority_class': majority_class,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """Fitting model to training data"""
        # Pre-calculate feature types to save time during split search
        self.feat_types = []
        for i in range(X.shape[1]):
            # Check if binary (0/1 only)
            unique = np.unique(X[:, i])
            # Remove nan
            unique = unique[~np.isnan(unique)]
            if len(unique) <= 2 and np.all(np.isin(unique, [0, 1])):
                self.feat_types.append('binary')
            else:
                self.feat_types.append('continuous')
                
        self.tree = self._build_tree(X, y, self.feat_types)
        return self

    def _predict_single(self, x, node):
        """Predict for single sample"""
        if node['leaf']:
            return node['value']
        
        feature_val = x[node['feature']]

        if np.isnan(feature_val):
            return node['majority_class'] 
        
        # Unified Split Logic
        if feature_val <= node['threshold']:
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