import cupy as np
import json # hapus ja save loadnya klo dipusatin bang

class SVMNode:
    def __init__(self, learning_rate=0.001, reg_strength=0.01, max_epochs=1000):
        self.lr = learning_rate
        self.reg = reg_strength 
        self.epochs = max_epochs
        self.weights = None
        self.bias = None

    def compute_margin(self, X):
        return np.dot(X, self.weights) + self.bias

    def train_node(self, X_train, y_train):
        num_samples, num_features = X_train.shape
        
        self.weights = np.zeros(num_features)
        self.bias = 0

        y_scaled = np.where(y_train <= 0, -1, 1)

        for epoch in range(self.epochs):
            indices = np.random.permutation(num_samples)
            
            for i in indices:
                xi = X_train[i]
                target = y_scaled[i]

                current_margin = self.compute_margin(xi)
                
                check_condition = target * current_margin >= 1
                
                if check_condition: # classified di luar margin
                    gradient_w = 2 * self.reg * self.weights
                    self.weights = self.weights - (self.lr * gradient_w)
                else:
                    gradient_w = (2 * self.reg * self.weights) - (target * xi)
                    self.weights = self.weights - (self.lr * gradient_w)

                    self.bias = self.bias + (self.lr * target)

class SVMScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        
        self.sub_classifiers = []
        self.known_classes = []

    def fit(self, features, targets): 
        # One-vs-All
        self.sub_classifiers = []
        self.known_classes = np.unique(targets)
        
        n_classes = len(self.known_classes)
        print(f"SVM untuk {n_classes} kelas: {self.known_classes}")
        
        for cls in self.known_classes:
            binary_labels = np.where(targets == cls, 1, -1)
            
            node = SVMNode(
                learning_rate=self.learning_rate,
                reg_strength=self.lambda_param,
                max_epochs=self.n_iters
            )

            node.train_node(features, binary_labels)

            self.sub_classifiers.append(node)

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

    def save_model(self, file_path):
        model_container = {
            "hyperparams": {
                "lr": self.learning_rate,
                "reg": self.lambda_param,
                "iters": self.n_iters
            },
            "classes": self.known_classes.tolist(),
            "weights_data": []
        }

        for node in self.sub_classifiers:
            node_data = {
                "w_list": node.weights.tolist(),
                "b_val": node.bias
            }
            model_container["weights_data"].append(node_data)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(model_container, f, indent=4)
            print(f"Model tersimpan di {file_path}")
        except Exception as e:
            print(f"Gagal menyimpan model: {e}")

    @staticmethod
    def load_model(file_path): # Load dri JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            hp = data["hyperparams"]
            new_svm = SVMScratch(
                learning_rate=hp["lr"], 
                lambda_param=hp["reg"], 
                n_iters=hp["iters"]
            )
            new_svm.known_classes = np.array(data["classes"])

            for w_data in data["weights_data"]:
                temp_node = SVMNode()
                temp_node.weights = np.array(w_data["w_list"])
                temp_node.bias = w_data["b_val"]
                temp_node.lr = hp["lr"]
                temp_node.reg = hp["reg"]
                temp_node.epochs = hp["iters"]

                new_svm.sub_classifiers.append(temp_node)
                
            print(f"Model berhasil dimuat dari {file_path}")
            return new_svm
            
        except FileNotFoundError:
            print("File tidak ditemukan.")
            return None
