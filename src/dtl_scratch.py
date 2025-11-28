import numpy as np
import pickle

class DecisionTreeScratch:
    """
    Enhanced DTL CART implementation with:
    - Gini or Entropy criterion
    - Post-pruning (Reduced Error Pruning)
    - Class weights for imbalanced data
    - Feature importance calculation
    - Better midpoint thresholds
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', min_impurity_decrease=0.0, class_weight=None,
                 max_features=None, random_state=None): 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion  # 'gini' or 'entropy'
        self.min_impurity_decrease = min_impurity_decrease  # Minimum gain to split
        self.class_weight = class_weight  # None, 'balanced', or dict
        self.max_features = max_features  # None, 'sqrt', 'log2', int, or float
        self.random_state = random_state
        self.tree = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.classes_ = None
        self._class_weights = None
        
        if random_state is not None:
            np.random.seed(random_state)

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        
        if self.class_weight is None:
            self._class_weights = {c: 1.0 for c in classes}
        elif self.class_weight == 'balanced':
            # Balanced: n_samples / (n_classes * count_per_class)
            n_samples = len(y)
            n_classes = len(classes)
            self._class_weights = {c: n_samples / (n_classes * count) 
                                   for c, count in zip(classes, counts)}
        elif isinstance(self.class_weight, dict):
            self._class_weights = self.class_weight
        else:
            self._class_weights = {c: 1.0 for c in classes}

    def _gini(self, y, sample_weights=None):
        """Calculate Gini impurity (weighted)"""
        n = len(y)
        if n == 0:
            return 0
        
        if sample_weights is None:
            counts = np.bincount(y.astype(int), minlength=len(self.classes_))
            total = n
        else:
            counts = np.bincount(y.astype(int), weights=sample_weights, 
                                 minlength=len(self.classes_))
            total = np.sum(sample_weights)
        
        if total == 0:
            return 0
            
        probs = counts / total
        return 1 - np.sum(probs ** 2)

    def _entropy(self, y, sample_weights=None):
        """Calculate Entropy (Information Gain criterion)"""
        n = len(y)
        if n == 0:
            return 0
        
        if sample_weights is None:
            counts = np.bincount(y.astype(int), minlength=len(self.classes_))
            total = n
        else:
            counts = np.bincount(y.astype(int), weights=sample_weights,
                                 minlength=len(self.classes_))
            total = np.sum(sample_weights)
        
        if total == 0:
            return 0
            
        probs = counts / total
        # Avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y, sample_weights=None):
        """Calculate impurity based on criterion"""
        if self.criterion == 'entropy':
            return self._entropy(y, sample_weights)
        return self._gini(y, sample_weights)

    def _get_sample_weights(self, y):
        """Get sample weights based on class weights"""
        return np.array([self._class_weights[int(label)] for label in y])

    def _split(self, X, y, feature_idx, threshold, sample_weights=None):
        """Split dataset (Vectorized)"""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if sample_weights is not None:
            return (X[left_mask], y[left_mask], sample_weights[left_mask],
                    X[right_mask], y[right_mask], sample_weights[right_mask])
        return (X[left_mask], y[left_mask], None,
                X[right_mask], y[right_mask], None)

    def _get_num_features_to_check(self, n_features):
        """Determine number of features to consider at each split"""
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return n_features

    def _best_split(self, X, y, feat_types, sample_weights=None):  
        """Finding best split with improved threshold selection"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_impurity = self._impurity(y, sample_weights)
        n_samples = len(y)
        n_features = X.shape[1]
        
        if sample_weights is not None:
            total_weight = np.sum(sample_weights)
        else:
            total_weight = n_samples
        
        # Feature subsampling for randomness (helps generalization)
        num_features_to_check = self._get_num_features_to_check(n_features)
        if num_features_to_check < n_features:
            features_to_check = np.random.choice(n_features, num_features_to_check, replace=False)
        else:
            features_to_check = range(n_features)
        
        for feature_idx in features_to_check:
            f_type = feat_types[feature_idx]
            
            if f_type == 'binary':
                thresholds = [0.5]
            else:
                feature_values = X[:, feature_idx]
                valid_mask = ~np.isnan(feature_values)
                sorted_unique = np.sort(np.unique(feature_values[valid_mask]))
                
                if len(sorted_unique) < 2:
                    continue
                
                # IMPROVEMENT: Use midpoints between consecutive values
                # This often gives better splits
                if len(sorted_unique) > 30:
                    # For many unique values, use percentile-based midpoints
                    percentiles = np.percentile(feature_values[valid_mask], 
                                                np.linspace(5, 95, 25))
                    thresholds = percentiles
                else:
                    # Use midpoints between consecutive sorted unique values
                    thresholds = (sorted_unique[:-1] + sorted_unique[1:]) / 2
            
            for threshold in thresholds:
                (X_left, y_left, sw_left, 
                 X_right, y_right, sw_right) = self._split(X, y, feature_idx, 
                                                           threshold, sample_weights)
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # Weighted impurity calculation
                if sample_weights is not None:
                    w_left = np.sum(sw_left) if sw_left is not None else len(y_left)
                    w_right = np.sum(sw_right) if sw_right is not None else len(y_right)
                else:
                    w_left, w_right = len(y_left), len(y_right)
                
                left_impurity = self._impurity(y_left, sw_left)
                right_impurity = self._impurity(y_right, sw_right)
                
                weighted_impurity = (w_left / total_weight) * left_impurity + \
                                    (w_right / total_weight) * right_impurity
                gain = parent_impurity - weighted_impurity
                
                # Apply minimum impurity decrease threshold
                if gain > best_gain and gain >= self.min_impurity_decrease:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def _weighted_majority_class(self, y, sample_weights=None):
        """Get majority class considering weights"""
        if sample_weights is None:
            counts = np.bincount(y.astype(int), minlength=len(self.classes_))
        else:
            counts = np.bincount(y.astype(int), weights=sample_weights,
                                 minlength=len(self.classes_))
        return np.argmax(counts)

    def _build_tree(self, X, y, feat_types, sample_weights=None, depth=0):
        """Building tree recursively with improvements"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        if n_samples == 0:
            return {'leaf': True, 'value': None, 'n_samples': 0}

        majority_class = self._weighted_majority_class(y, sample_weights)
        
        # Store class distribution for probability predictions
        if sample_weights is None:
            class_counts = np.bincount(y.astype(int), minlength=len(self.classes_))
        else:
            class_counts = np.bincount(y.astype(int), weights=sample_weights,
                                       minlength=len(self.classes_))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split or \
           n_samples < 2 * self.min_samples_leaf:
            return {'leaf': True, 'value': majority_class, 'n_samples': n_samples,
                    'class_counts': class_counts}
        
        feature_idx, threshold, gain = self._best_split(X, y, feat_types, sample_weights)
        
        if feature_idx is None:
            return {'leaf': True, 'value': majority_class, 'n_samples': n_samples,
                    'class_counts': class_counts}
        
        (X_left, y_left, sw_left, 
         X_right, y_right, sw_right) = self._split(X, y, feature_idx, threshold, sample_weights)
        
        left_subtree = self._build_tree(X_left, y_left, feat_types, sw_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, feat_types, sw_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'gain': gain,
            'majority_class': majority_class,
            'n_samples': n_samples,
            'class_counts': class_counts,
            'left': left_subtree,
            'right': right_subtree
        }

    def _prune_tree(self, node, X_val, y_val):
        """
        Reduced Error Pruning (REP):
        Replace a subtree with a leaf if it doesn't decrease validation accuracy.
        """
        if node['leaf']:
            return node
        
        # Recursively prune children first (bottom-up)
        node['left'] = self._prune_tree(node['left'], X_val, y_val)
        node['right'] = self._prune_tree(node['right'], X_val, y_val)
        
        # Get predictions with current subtree
        preds_with_subtree = self._predict_node(X_val, node)
        acc_with_subtree = np.mean(preds_with_subtree == y_val)
        
        # Get predictions if we replace with leaf
        preds_as_leaf = np.full(len(y_val), node['majority_class'])
        acc_as_leaf = np.mean(preds_as_leaf == y_val)
        
        # If pruning doesn't hurt (or helps), prune
        if acc_as_leaf >= acc_with_subtree:
            return {
                'leaf': True,
                'value': node['majority_class'],
                'n_samples': node['n_samples'],
                'class_counts': node.get('class_counts', None)
            }
        
        return node

    def _predict_node(self, X, node):
        """Predict using a specific node as root"""
        return np.array([self._predict_single(x, node) for x in X])

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit model to training data.
        If X_val and y_val provided, perform post-pruning.
        """
        self.n_features_ = X.shape[1]
        self._compute_class_weights(y)
        
        # Get sample weights
        sample_weights = self._get_sample_weights(y)
        
        # Pre-calculate feature types
        self.feat_types = []
        for i in range(X.shape[1]):
            unique = np.unique(X[:, i])
            unique = unique[~np.isnan(unique)]
            if len(unique) <= 2 and np.all(np.isin(unique, [0, 1])):
                self.feat_types.append('binary')
            else:
                self.feat_types.append('continuous')
        
        # Build tree
        self.tree = self._build_tree(X, y, self.feat_types, sample_weights)
        
        # Post-pruning if validation set provided
        if X_val is not None and y_val is not None:
            self.tree = self._prune_tree(self.tree, X_val, y_val)
        
        # Calculate feature importances
        self._compute_feature_importances()
        
        return self

    def _compute_feature_importances(self):
        """Compute feature importances based on total gain"""
        importances = np.zeros(self.n_features_)
        total_samples = self.tree.get('n_samples', 1)
        
        def traverse(node):
            if node['leaf']:
                return
            
            feature = node['feature']
            gain = node.get('gain', 0)
            n_samples = node.get('n_samples', 1)
            
            # Weight importance by number of samples reaching this node
            importances[feature] += gain * (n_samples / total_samples)
            
            traverse(node['left'])
            traverse(node['right'])
        
        traverse(self.tree)
        
        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        
        self.feature_importances_ = importances

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

    def _predict_proba_single(self, x, node):
        """Get class probabilities for single sample"""
        if node['leaf']:
            counts = node.get('class_counts', None)
            if counts is None:
                # Fallback: one-hot for the predicted class
                probs = np.zeros(len(self.classes_))
                if node['value'] is not None:
                    probs[node['value']] = 1.0
                return probs
            total = np.sum(counts)
            if total == 0:
                return np.ones(len(self.classes_)) / len(self.classes_)
            return counts / total
        
        feature_val = x[node['feature']]

        if np.isnan(feature_val):
            # Use class distribution at this node
            counts = node.get('class_counts', None)
            if counts is not None:
                return counts / np.sum(counts)
            probs = np.zeros(len(self.classes_))
            probs[node['majority_class']] = 1.0
            return probs
        
        if feature_val <= node['threshold']:
            return self._predict_proba_single(x, node['left'])
        else:
            return self._predict_proba_single(x, node['right'])

    def predict(self, X):
        """Predict target values"""
        return np.array([self._predict_single(x, self.tree) for x in X])

    def predict_proba(self, X):
        """Predict class probabilities"""
        return np.array([self._predict_proba_single(x, self.tree) for x in X])

    def get_depth(self, node=None):
        """Get the depth of the tree"""
        if node is None:
            node = self.tree
        if node['leaf']:
            return 0
        return 1 + max(self.get_depth(node['left']), self.get_depth(node['right']))

    def get_n_leaves(self, node=None):
        """Get the number of leaves in the tree"""
        if node is None:
            node = self.tree
        if node['leaf']:
            return 1
        return self.get_n_leaves(node['left']) + self.get_n_leaves(node['right'])

    def save(self, filepath):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)