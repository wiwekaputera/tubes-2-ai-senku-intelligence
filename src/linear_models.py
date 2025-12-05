import numpy as np
import pickle

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm


class LogisticRegression:
    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        batch_size=32,
        lambda_reg=0.01,
        decay_rate=0.0,
    ):
        """
        Logistic Regression (mini-batch GD) + L2 regularization + learning-rate decay.

        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.decay_rate = decay_rate
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.theta_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        positive_mask = z >= 0
        result = np.zeros_like(z, dtype=np.float64)
        result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
        exp_z = np.exp(z[~positive_mask])
        result[~positive_mask] = exp_z / (1.0 + exp_z)
        return result

    def _compute_loss(self, X, y):
        n = len(y)
        z = np.dot(X, self.weights) + self.bias
        p = self._sigmoid(z)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        # Regularisasi L2, bias tidak termasuk
        l2_term = (self.lambda_reg / (2 * n)) * np.sum(self.weights**2)
        return bce + l2_term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0.0
        self.loss_history = []
        self.theta_history = []  # Simpan (bias, w1) untuk visualisasi kontur
        batch_size = (
            n_samples if self.batch_size == -1 else min(self.batch_size, n_samples)
        )
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(self.n_iterations):
            current_lr = self.learning_rate / (1.0 + self.decay_rate * epoch)
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                batch_len = len(y_batch)
                z = np.dot(X_batch, self.weights) + self.bias
                p = self._sigmoid(z)

                error = p - y_batch
                grad_w = (1.0 / batch_len) * np.dot(X_batch.T, error)
                grad_w += self.lambda_reg * self.weights
                grad_b = np.mean(error)

                self.weights -= current_lr * grad_w
                self.bias -= current_lr * grad_b

            # Proyeksi (bias, w1) untuk visualisasi kontur
            if self.weights is not None and self.weights.size > 0:
                self.theta_history.append((float(self.bias), float(self.weights[0])))

            if epoch % 50 == 0 or epoch == self.n_iterations - 1:
                loss = self._compute_loss(X, y)
                self.loss_history.append((epoch, loss))

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
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
            if hasattr(model, "loss_history"):
                histories[f"class_{cls}"] = model.loss_history
        return histories

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


# fucntion for contour visualization
def sliced_contour_loss(theta0, theta1, X_full, y, lambda_reg, theta_fixed):
    # method slicing hanya theta0 dan theta1 digerakkan, theta lain dikunci pada nilai final
    w_full = np.insert(theta_fixed, 0, theta1)
    z = theta0 + X_full @ w_full
    p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    l2_term = (lambda_reg / (2 * len(y))) * np.sum(w_full**2)
    return bce + l2_term


def visualize_contour(model_full, X_full, y, out_gif_path, grid_size=60, fps=8):
    is_ova = isinstance(model_full, OneVsAll)

    if is_ova:
        sub_model = model_full.models[0]
        theta_final = np.insert(sub_model.weights.flatten(), 0, sub_model.bias)
        lambda_reg = sub_model.lambda_reg
    else:
        sub_model = model_full
        theta_final = np.insert(model_full.weights.flatten(), 0, model_full.bias)
        lambda_reg = getattr(model_full, "lambda_reg", 0.01)

    theta0_final = float(theta_final[0])
    theta1_final = float(theta_final[1])
    theta_fixed = theta_final[2:]

    t0 = np.linspace(theta0_final - 2, theta0_final + 2, grid_size)
    t1 = np.linspace(theta1_final - 1, theta1_final + 1, grid_size)
    T0, T1 = np.meshgrid(t0, t1)

    # hitung hasil slicing loss Z
    Z = np.array(
        [
            sliced_contour_loss(t0i, t1j, X_full, y, lambda_reg, theta_fixed)
            for t0i, t1j in zip(np.ravel(T0), np.ravel(T1))
        ]
    ).reshape(T0.shape)

    loss_hist = list(getattr(sub_model, "loss_history", []))
    theta_path = np.array(getattr(sub_model, "theta_history", []), dtype=float)
    has_path = len(theta_path) > 0 and theta_path.ndim == 2 and theta_path.shape[1] == 2
    has_loss = len(loss_hist) > 0
    loss_epochs = (
        np.array([int(ep) for ep, _ in loss_hist]) if has_loss else np.array([])
    )

    def path_slice_idx(frame_idx):
        if has_loss and has_path and frame_idx < len(loss_epochs):
            idx = loss_epochs[frame_idx]
            return min(idx, len(theta_path) - 1)
        return len(theta_path) - 1

    if has_loss:
        final_epoch = int(loss_hist[-1][0])
        final_loss = float(loss_hist[-1][1])
    else:
        final_epoch, final_loss = None, None

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(
        " Visualisasi garis kontur fungsi loss (log-loss) dan lintasan parameter (θ₀, θ₁) selama training."
    )
    ax.set_xlabel(r"$\theta_0$ (Bias)")
    ax.set_ylabel(r"$\theta_1$ (Weight F1)")

    levels = np.logspace(np.log10(max(Z.min() * 1.05, 1e-6)), np.log10(Z.max()), 12)
    CS = ax.contour(T0, T1, Z, levels=levels, cmap=cm.viridis, linewidths=0.7)
    ax.clabel(CS, inline=1, fontsize=8, fmt="%1.2f")

    (line,) = ax.plot(
        [], [], "r-", marker="o", markersize=4, label="Proyeksi Lintasan (bias,w1)"
    )
    (point,) = ax.plot([], [], "ro", markersize=8, label="Posisi Saat Ini", alpha=0.9)
    (optpt,) = ax.plot(
        [theta0_final], [theta1_final], "ro", markersize=8, label="Optimum", alpha=0.4
    )
    text_info = ax.text(0.05, 0.95, "", transform=ax.transAxes)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        if has_loss:
            text_info.set_text(
                f"Epoch: {int(loss_hist[0][0])}\nLoss: {float(loss_hist[0][1]):.4f}"
            )
        else:
            text_info.set_text("Epoch: 0")
        ax.legend(loc="upper right")
        return line, point, optpt, text_info

    def update(frame):
        if has_path:
            end_idx = path_slice_idx(frame)
            t0_path = theta_path[: end_idx + 1, 0]
            t1_path = theta_path[: end_idx + 1, 1]
            line.set_data(t0_path, t1_path)
            point.set_data([t0_path[-1]], [t1_path[-1]])
        if has_loss:
            ep, ls = loss_hist[frame]
            text_info.set_text(f"Epoch: {int(ep)}\nLoss: {float(ls):.4f}")
        return line, point, optpt, text_info

    frames = len(loss_hist) if has_loss else (len(theta_path) if has_path else 1)
    ani = FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=200
    )

    project_root = os.path.dirname(os.path.dirname(__file__))
    os.makedirs(os.path.join(project_root, "images"), exist_ok=True)
    ani.save(out_gif_path, writer="pillow", fps=fps)
    plt.close(fig)

    if final_epoch is not None:
        print(
            f"GIF disimpan: {out_gif_path} \n Epoch={final_epoch}, Loss={final_loss:.6f}"
        )
