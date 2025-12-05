import os
import numpy as np
from datetime import datetime, timezone

from linear_models import LogisticRegression, OneVsAll, visualize_contour


if os.path.exists("data/processed/X_train.npy"):

    data_path = "data/processed"
    images_path = "images"
elif os.path.exists("../data/processed/X_train.npy"):
    data_path = "../data/processed"
    images_path = "../images"
else:
    raise FileNotFoundError("Cannot find data")

# Load processed arrays
X_train = np.load(os.path.join(data_path, "X_train.npy"))
y_train = np.load(os.path.join(data_path, "y_train.npy"))
X_val   = np.load(os.path.join(data_path, "X_val.npy"))
y_val   = np.load(os.path.join(data_path, "y_val.npy"))

print("Data successfully loaded!")

# quick parameter for a fast test
config = {
    'model_class': LogisticRegression,
    'learning_rate': 0.1,
    'n_iterations': 500,
    'batch_size': 32,
    'lambda_reg': 0.1,
    'decay_rate': 0.001,
}

print("Using LogReg OvA test configs:")
print(config)
print("="*60)

# Train LogReg OvA
print("Training LogReg OvA...")
lr_ova = OneVsAll(**config)
lr_ova.fit(X_train, y_train)

# predict and evaluate
train_pred = lr_ova.predict(X_train)
val_pred   = lr_ova.predict(X_val)
train_acc = np.mean(train_pred == y_train)
val_acc   = np.mean(val_pred == y_val)

print(f"LOGREG TRAIN ACC: {train_acc * 100:.2f}%")
print(f"LOGREG VAL   ACC: {val_acc * 100:.2f}%")
print("="*60)

os.makedirs(images_path, exist_ok=True)

# Generate contour visualization GIF
print("Generating contour visualization GIF...")
date_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
gif_path = os.path.join(images_path, f"logreg_test_{date_str}.gif")

X_viz, y_viz = X_val, y_val

visualize_contour(lr_ova, X_viz, y_viz, gif_path, grid_size=60, fps=8)

print("\n" + "="*60)
print("Contour Visualization Test Completed")
print("="*60)
print(f"Output GIF saved to: {gif_path}")
