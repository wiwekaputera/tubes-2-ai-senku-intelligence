import numpy as np
import pickle

class DecisionTreeScratch:
    """
    DTL CART implementation with gini/entropy criterion, also supports pruning
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', min_impurity_decrease=0.0, class_weight=None,
                 max_features=None, random_state=None): 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None
        self.n_features_ = None
        self.feature_importances_ = None
        self.classes_ = None
        self._class_weights = None
        
        if random_state is not None:
            np.random.seed(random_state)

    def _compute_class_weights(self, y):
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        
        if self.class_weight is None:
            self._class_weights = {c: 1.0 for c in classes}
        elif self.class_weight == 'balanced':
            n_samples = len(y)
            n_classes = len(classes)
            self._class_weights = {c: n_samples / (n_classes * count) 
                                   for c, count in zip(classes, counts)}
        elif isinstance(self.class_weight, dict):
            self._class_weights = self.class_weight
        else:
            self._class_weights = {c: 1.0 for c in classes}

    # Calculate gini impurity
    def _gini(self, y, sample_weights=None):
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

    # Calculate entropy scorenya
    def _entropy(self, y, sample_weights=None):
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
        probs = probs[probs > 0]  # avoid log(0)
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y, sample_weights=None):
        if self.criterion == 'entropy':
            return self._entropy(y, sample_weights)
        return self._gini(y, sample_weights)

    def _get_sample_weights(self, y):
        return np.array([self._class_weights[int(label)] for label in y])

    def _split(self, X, y, feature_idx, threshold, sample_weights=None):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if sample_weights is not None:
            return (X[left_mask], y[left_mask], sample_weights[left_mask],
                    X[right_mask], y[right_mask], sample_weights[right_mask])
        return (X[left_mask], y[left_mask], None,
                X[right_mask], y[right_mask], None)

    def _get_num_features_to_check(self, n_features):
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

    # Look for best split
    def _best_split(self, X, y, feat_types, sample_weights=None):
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
                
                # Get candidate thresholds
                if len(sorted_unique) > 30:
                    percentiles = np.percentile(feature_values[valid_mask], 
                                                np.linspace(5, 95, 25))
                    thresholds = percentiles
                else:
                    thresholds = (sorted_unique[:-1] + sorted_unique[1:]) / 2
            
            # Try each threshold 
            for threshold in thresholds:
                # Split and check min leaf size
                (X_left, y_left, sw_left, 
                 X_right, y_right, sw_right) = self._split(X, y, feature_idx, 
                                                           threshold, sample_weights)
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                if sample_weights is not None:
                    w_left = np.sum(sw_left) if sw_left is not None else len(y_left)
                    w_right = np.sum(sw_right) if sw_right is not None else len(y_right)
                else:
                    w_left, w_right = len(y_left), len(y_right)
                
                left_impurity = self._impurity(y_left, sw_left)
                right_impurity = self._impurity(y_right, sw_right)
                
                # Find weighted average impuritynya
                weighted_impurity = (w_left / total_weight) * left_impurity + \
                                    (w_right / total_weight) * right_impurity
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain and gain >= self.min_impurity_decrease:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def _weighted_majority_class(self, y, sample_weights=None):
        if sample_weights is None:
            counts = np.bincount(y.astype(int), minlength=len(self.classes_))
        else:
            counts = np.bincount(y.astype(int), weights=sample_weights,
                                 minlength=len(self.classes_))
        return np.argmax(counts)

    # Build tree
    def _build_tree(self, X, y, feat_types, sample_weights=None, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        if n_samples == 0:
            return {'leaf': True, 'value': None, 'n_samples': 0}

        majority_class = self._weighted_majority_class(y, sample_weights)
        
        if sample_weights is None:
            class_counts = np.bincount(y.astype(int), minlength=len(self.classes_))
        else:
            class_counts = np.bincount(y.astype(int), weights=sample_weights,
                                       minlength=len(self.classes_))
        
        # Stop splitting if hit stop conditions
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
        
        # Save node infoo
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
    
    # Pruning
    def _prune_tree(self, node, X_val, y_val):
        if node['leaf']:
            return node
        
        node['left'] = self._prune_tree(node['left'], X_val, y_val)
        node['right'] = self._prune_tree(node['right'], X_val, y_val)
        
        preds_with_subtree = self._predict_node(X_val, node)
        acc_with_subtree = np.mean(preds_with_subtree == y_val)
        
        preds_as_leaf = np.full(len(y_val), node['majority_class'])
        acc_as_leaf = np.mean(preds_as_leaf == y_val)
        
        # Prune and make subtree into a majority class node
        if acc_as_leaf >= acc_with_subtree:
            return {
                'leaf': True,
                'value': node['majority_class'],
                'n_samples': node['n_samples'],
                'class_counts': node.get('class_counts', None)
            }
        
        return node

    def _predict_node(self, X, node):
        return np.array([self._predict_single(x, node) for x in X])

    # Setup for training
    def fit(self, X, y, X_val=None, y_val=None):
        self.n_features_ = X.shape[1]
        self._compute_class_weights(y)
        
        sample_weights = self._get_sample_weights(y)
        
        # Find feature type
        self.feat_types = []
        for i in range(X.shape[1]):
            unique = np.unique(X[:, i])
            unique = unique[~np.isnan(unique)]
            if len(unique) <= 2 and np.all(np.isin(unique, [0, 1])):
                self.feat_types.append('binary')
            else:
                self.feat_types.append('continuous')
        
        self.tree = self._build_tree(X, y, self.feat_types, sample_weights)
        
        if X_val is not None and y_val is not None:
            self.tree = self._prune_tree(self.tree, X_val, y_val)
        
        self._compute_feature_importances()
        
        return self

    # Calculate feature importance
    def _compute_feature_importances(self):
        importances = np.zeros(self.n_features_)
        total_samples = self.tree.get('n_samples', 1)
        
        def traverse(node):
            if node['leaf']:
                return
            feature = node['feature']
            gain = node.get('gain', 0)
            n_samples = node.get('n_samples', 1)
            importances[feature] += gain * (n_samples / total_samples)
            traverse(node['left'])
            traverse(node['right'])
        
        traverse(self.tree)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        self.feature_importances_ = importances

    # Traverse
    def _predict_single(self, x, node):
        if node['leaf']:
            return node['value']
        
        feature_val = x[node['feature']]

        if np.isnan(feature_val):
            return node['majority_class'] 
        
        if feature_val <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    # Predict probsnya
    def _predict_proba_single(self, x, node):
        if node['leaf']:
            counts = node.get('class_counts', None)
            if counts is None:
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
        return np.array([self._predict_single(x, self.tree) for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_proba_single(x, self.tree) for x in X])

    def get_depth(self, node=None):
        if node is None:
            node = self.tree
        if node['leaf']:
            return 0
        return 1 + max(self.get_depth(node['left']), self.get_depth(node['right']))

    def get_n_leaves(self, node=None):
        if node is None:
            node = self.tree
        if node['leaf']:
            return 1
        return self.get_n_leaves(node['left']) + self.get_n_leaves(node['right'])

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    # Bonus: Tree Visualization
    def visualize_tree(self, max_depth=3, save_path='tree.png'):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except:
            print("need matplotlib")
            return
        
        def limit(node, d, curr=0):
            if node['leaf'] or curr >= d:
                return {'leaf': True, 'value': node.get('value', 0), 'n_samples': node.get('n_samples', 0)}
            return {'leaf': False, 'feature': node['feature'], 'threshold': node['threshold'],
                   'n_samples': node.get('n_samples', 0),
                   'left': limit(node['left'], d, curr+1), 'right': limit(node['right'], d, curr+1)}
        
        def draw(ax, node, x, y, w, positions):
            positions[id(node)] = (x, y)
            if node['leaf']:
                ax.add_patch(Rectangle((x-0.25, y-0.12), 0.5, 0.24, fc='lightcyan', ec='black', lw=1.5))
                ax.text(x, y, f"Class {node['value']}\nn={node['n_samples']}", ha='center', va='center', fontsize=9)
            else:
                ax.add_patch(Rectangle((x-0.25, y-0.12), 0.5, 0.24, fc='lightblue', ec='black', lw=1.5))
                ax.text(x, y, f"X[{node['feature']}] <= {node['threshold']:.2f}\nn={node['n_samples']}", 
                       ha='center', va='center', fontsize=9)
                draw(ax, node['left'], x-w/2, y-1.2, w/2, positions)
                draw(ax, node['right'], x+w/2, y-1.2, w/2, positions)
                x_l, y_l = positions[id(node['left'])]
                x_r, y_r = positions[id(node['right'])]
                ax.plot([x, x_l], [y-0.12, y_l+0.12], 'k-', lw=1)
                ax.plot([x, x_r], [y-0.12, y_r+0.12], 'k-', lw=1)
        
        tree = limit(self.tree, max_depth)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        positions = {}
        draw(ax, tree, 0, 0, 2.5, positions)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-max_depth*1.3, 0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Tree has been saved in: {save_path}")
    
    # Best 10 top splits are saved
    def visualize_top_branches(self, n=10, save_path='top_branches.png'):
        try:
            import matplotlib.pyplot as plt
        except:
            print("Matplotlib not installed")
            return
        
        nodes = []
        def collect(node, d=0):
            if not node['leaf']:
                nodes.append((node.get('gain', 0), d, node))
                collect(node['left'], d+1)
                collect(node['right'], d+1)
        
        collect(self.tree)
        nodes.sort(reverse=True, key=lambda x: x[0])
        
        data = [[f"#{i+1}", f"X[{g[2]['feature']}]", f"{g[2]['threshold']:.2f}", 
                f"{g[0]:.4f}", g[2].get('n_samples', 0), g[1]] for i, g in enumerate(nodes[:n])]
        
        fig, ax = plt.subplots(figsize=(12, max(5, n*0.35)))
        ax.axis('off')
        
        t = ax.table(cellText=data, colLabels=['Rank', 'Feature', 'Threshold', 'Gain', 'Samples', 'Depth'],
                    cellLoc='center', loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.8)
        
        for i in range(6):
            t[(0, i)].set_facecolor("#C1FFFD")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Tree has been saved in: {save_path}")

# For testing 
if __name__ == "__main__":
    import os
    
    print("Loading data...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    
    print("Training...")
    model = DecisionTreeScratch(max_depth=7, min_samples_split=8, min_samples_leaf=4)
    model.fit(X_train, y_train)
    
    train_acc = np.mean(model.predict(X_train) == y_train)
    val_acc = np.mean(model.predict(X_val) == y_val)
    print(f"Train acc: {train_acc*100:.2f}%")
    print(f"Val acc: {val_acc*100:.2f}%")
    
    print("\nGenerating Tree Visualization...")
    os.makedirs("doc", exist_ok=True)
    
    model.visualize_tree(max_depth=3, save_path='doc/tree_depth3.png')
    model.visualize_top_branches(n=10, save_path='doc/tree_top10.png')
    
    print("Finished!")