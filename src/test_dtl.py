import numpy as np
import os
from dtl_scratch import DecisionTreeScratch

# Load datanya
print("Loading Data...")

# Deteksi apakah dijalankan dari src/ atau root
if os.path.exists("data/processed/X_train.npy"):
    # Dijalankan dari root project
    data_path = "data/processed"
    models_path = "models"
    doc_path = "doc"
elif os.path.exists("../data/processed/X_train.npy"):
    # Dijalankan dari src/
    data_path = "../data/processed"
    models_path = "../models"
    doc_path = "../doc"
else:
    raise FileNotFoundError("Cannot find data/processed/ folder. Please run from project root or src/ folder.")

X_train = np.load(f"{data_path}/X_train.npy")
y_train = np.load(f"{data_path}/y_train.npy")
X_val   = np.load(f"{data_path}/X_val.npy")
y_val   = np.load(f"{data_path}/y_val.npy")

print("Data successfully loaded!")
print(f"Train shape: {X_train.shape}, y={y_train.shape}")
print(f"Val   shape: {X_val.shape}, y={y_val.shape}")
print("="*60)

# Optimal hyperparameters
config = {
    'max_depth': 7,
    'min_samples_split': 8,
    'min_samples_leaf': 4,
    'criterion': 'gini'
}

print("Use best configs:")
print(config)
print("="*60)

# Train CART
print("Training CART...")
cart = DecisionTreeScratch(**config)
cart.fit(X_train, y_train)

train_pred = cart.predict(X_train)
val_pred   = cart.predict(X_val)

train_acc = np.mean(train_pred == y_train)
val_acc   = np.mean(val_pred == y_val)

print(f"CART TRAIN ACC: {train_acc * 100:.2f}%")
print(f"CART VAL   ACC: {val_acc * 100:.2f}%")
print("="*60)

# Test fungsi save dan load
print("Testing Save/Load...")

# Create models directory if not exists
os.makedirs(models_path, exist_ok=True)

cart.save(f"{models_path}/cart_best.pkl")
loaded_cart = DecisionTreeScratch.load(f"{models_path}/cart_best.pkl")

loaded_val_pred = loaded_cart.predict(X_val)
loaded_val_acc = np.mean(loaded_val_pred == y_val)

print(f"Loaded CART VAL ACC: {loaded_val_acc * 100:.2f}%")
print("="*60)

# Tree stats
print("Tree Statistics:")
print(f"Tree Depth: {cart.get_depth()}")
print(f"Number of Leaves: {cart.get_n_leaves()}")
print("="*60)

# Buat testing bonus tree visualization
print("\nTree Visualization...")
os.makedirs(doc_path, exist_ok=True)

try:
    # Tree structure (depth 3)
    print("1. Generating tree visualization (depth 3)...")
    cart.visualize_tree(
        max_depth=3, 
        save_path=f'{doc_path}/tree_depth3.png'
    )
    
    # Tree structure (depth 5)
    print("2. Generating tree visualization (depth 5)...")
    cart.visualize_tree(
        max_depth=5, 
        save_path=f'{doc_path}/tree_depth5.png'
    )
    
    # Top-10 branches 
    print("3. Generating top-10 branches visualization...")
    cart.visualize_top_branches(
        n=10, 
        save_path=f'{doc_path}/tree_top10.png'
    )
    
    # Top-20 branches
    print("4. Generating top-20 branches visualization...")
    cart.visualize_top_branches(
        n=20, 
        save_path=f'{doc_path}/tree_top20.png'
    )
    
    print("\n" + "="*60)
    print("All Visualization have been created successfully")
    print("="*60)
    print(f"\nOutput files in '{doc_path}/' folder:")
    print("="*60)
    
except AttributeError:
    print("\nWARNING: Visualization methods not found!")
except Exception as e:
    print(f"\nError creating visualizations: {e}")
    print("Make sure matplotlib is installed")

print("\nCART testing completed!")