import sys
import os
import numpy as np
import itertools
import pandas as pd

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import run_full_pipeline as run_pipeline
from src.dtl_scratch import DecisionTreeScratch
from src.linear_models import LogisticRegressionScratch, OneVsAllClassifier
from src.svm_scratch import SVMScratch

def load_processed_data():
    try:
        base_path = os.path.join(os.path.dirname(__file__), 'data', 'processed')
        X_train = np.load(os.path.join(base_path, 'X_train.npy'))
        y_train = np.load(os.path.join(base_path, 'y_train.npy'))
        X_test_kaggle = np.load(os.path.join(base_path, 'X_test_kaggle.npy'))
        test_ids = np.load(os.path.join(base_path, 'test_ids.npy'))
        print(f"Data Loaded. Train shape: {X_train.shape}")
        return X_train, y_train, X_test_kaggle, test_ids
    except FileNotFoundError:
        print("Error: Processed data not found. Run preprocessing pipeline first.")
        sys.exit(1)

def random_oversampling(X, y):
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    
    X_balanced = []
    y_balanced = []
    
    for cls in unique_classes:
        # Get indices for this class
        indices = np.where(y == cls)[0]
        
        # Select random samples with replacement until we reach max_count
        oversampled_indices = np.random.choice(indices, size=max_count, replace=True)
        
        X_balanced.append(X[oversampled_indices])
        y_balanced.append(y[oversampled_indices])
        
    # Concatenate and Shuffle
    X_res = np.vstack(X_balanced)
    y_res = np.hstack(y_balanced)
    
    # Shuffle to mix the classes
    perm = np.random.permutation(len(X_res))
    return X_res[perm], y_res[perm]

def k_fold_cross_validation(model_class, params, X, y, k=5):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    accuracies = []
    
    for i in range(k):
        # 1. SPLIT (Raw, Imbalanced Data)
        start = i * fold_size
        end = (i + 1) * fold_size
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        
        X_fold_train_raw, y_fold_train_raw = X[train_idx], y[train_idx]
        X_fold_val, y_fold_val = X[val_idx], y[val_idx]
        
        # 2. OVERSAMPLE (Only the Training Portion!)
        # We reuse your existing random_oversampling function here
        X_fold_train_bal, y_fold_train_bal = random_oversampling(X_fold_train_raw, y_fold_train_raw)
        
        # 3. TRAIN (On Balanced Data)
        model = model_class(**params)
        model.fit(X_fold_train_bal, y_fold_train_bal)
        
        # 4. VALIDATE (On Original, Imbalanced Validation Data)
        preds = model.predict(X_fold_val)
        
        # Calculate Accuracy
        acc = np.mean(preds == y_fold_val)
        accuracies.append(acc)
        
    return np.mean(accuracies)

def grid_search(model_class, param_grid, X, y, k=5):
    """
    Exhaustively searches over param_grid to find best parameters.
    """
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    best_score = -1
    best_params = None
    
    print(f"Starting Grid Search for {model_class.__name__} ({len(combinations)} combos)...")
    
    for combo in combinations:
        # Create a dictionary for this specific combination
        # e.g., {'max_depth': 5, 'min_samples': 10}
        params = dict(zip(keys, combo))
        
        try:
            score = k_fold_cross_validation(model_class, params, X, y, k=k)
            # print(f"   Tested {params} -> Acc: {score:.4f}") # Optional: Verbose
            
            if score > best_score:
                best_score = score
                best_params = params
        except Exception as e:
            print(f"Crash with params {params}: {e}")
            
    print(f"Best Params: {best_params} | Best CV Score: {best_score:.4f}")
    return best_params, best_score


def main():
    print("Checking project structure...")
    
    # 1. Verify Preprocessing
    print("\n[1] Preprocessing Module: OK")
    run_pipeline()

    # 2. Verify Models
    dt = DecisionTreeScratch()
    lr = LogisticRegressionScratch()
    ova = OneVsAllClassifier(LogisticRegressionScratch)
    svm = SVMScratch()
    
    print(f"[2] Models Loaded Successfully:")
    print(f"    - {dt.__class__.__name__}")
    print(f"    - {lr.__class__.__name__}")
    print(f"    - {ova.__class__.__name__}")
    print(f"    - {svm.__class__.__name__}")

    print("\nProject scaffolding is ready.")

# ==========================================
    
    # 1. Load Data
    X_train, y_train, X_test_kaggle, test_ids = load_processed_data()
    
    # 2. Handle Imbalance (Crucial for scratching algorithms!)
    print(f"[3] Original Class Distribution: {np.unique(y_train, return_counts=True)}")
    X_train_bal, y_train_bal = random_oversampling(X_train, y_train)
    print(f"[4] Balanced Class Distribution: {np.unique(y_train_bal, return_counts=True)}")
    
    # 3. Define Grids for each model
    
    # --- Model A: Decision Tree (Native Multiclass) ---
    # Adjust 'max_depth' based on your tree implementation details
    dtl_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # --- Model B: Logistic Regression (Wrapped in OneVsAll) ---
    # We pass 'LogisticRegressionScratch' as a parameter to OvA (masi placeholder nunggu lrnya beres)
    logreg_ova_grid = {
        'model_class': [LogisticRegressionScratch],  # MUST be a list with one item
        'learning_rate': [0.1, 0.01, 0.001],
        'epochs': [500, 1000]
    }
    
    # --- Model C: SVM ---
    svm_grid = {
        'learning_rate': [0.001, 0.0001],
        'lambda_param': [0.01, 0.1, 1.0],
        'n_iters': [500, 1000]
    }

    # 4. Run Grid Search
    print("[4] Tuning")
    print("\n--- Tuning Decision Tree ---")
    best_dtl_params, best_dtl_score = grid_search(DecisionTreeScratch, dtl_grid, X_train, y_train)

    print("\n--- Tuning Logistic Regression (OvA) ---")
    best_lr_params, best_lr_score = grid_search(OneVsAllClassifier, logreg_ova_grid, X_train, y_train)

    print("\n--- Tuning SVM (OvA) ---")
    best_svm_params, best_svm_score = grid_search(SVMScratch, svm_grid, X_train, y_train)

    # 5. Final Training & Kaggle Submission
    # We select the best model based on CV score
    scores = {
        "DTL": best_dtl_score,
        "LogReg": best_lr_score,
        "SVM": best_svm_score
    }
    winner_name = max(scores, key=scores.get)
    print(f"\nWinning Model: {winner_name} (Acc: {scores[winner_name]:.4f})")
    
    # Retrain winner on FULL dataset (Balanced)
    final_model = None
    if winner_name == "DTL":
        final_model = DecisionTreeScratch(**best_dtl_params)
    elif winner_name == "LogReg":
        final_model = OneVsAllClassifier(**best_lr_params)
    elif winner_name == "SVM":
        final_model = SVMScratch(**best_svm_params)
        
    final_model.fit(X_train_bal, y_train_bal)
    
    # Predict on Kaggle Test Set
    numeric_preds = final_model.predict(X_test_kaggle)

    reverse_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    
    string_preds = [reverse_map[pred] for pred in numeric_preds]
    
    # Save CSV
    submission_df = pd.DataFrame({
        'Student_ID': test_ids,
        'Target': string_preds # Ensure this matches the required column name (e.g. 'Target' or 'Status')
    })
    
    # Map numeric predictions back to strings if necessary 
    # (Check if P1's encoder map is available, otherwise submit numeric if allowed)
    
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")

if __name__ == "__main__":
    main()
