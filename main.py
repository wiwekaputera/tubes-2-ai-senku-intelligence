import sys
import os
import numpy as np
import itertools
import pandas as pd
import pickle
import argparse
from joblib import Parallel, delayed
from datetime import datetime, timezone

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import run_full_pipeline as run_pipeline, SMOTEScratch
from src.dtl_scratch import DecisionTreeScratch
from src.linear_models import LogisticRegression, OneVsAll
from src.svm_scratch import SVMScratch

# ============ MODEL PERSISTENCE ============

def save_model(model, model_name, params, score, base_path="models"):
    """
    Save trained model with metadata.
    Uses pickle (allowed per project spec).
    """
    os.makedirs(base_path, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(base_path, filename)
    
    model_data = {
        'model': model,
        'name': model_name,
        'params': params,
        'score': score,
        'timestamp': timestamp,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved: {filepath}")
    return filepath


def load_model(filepath):
    """Load a saved model with metadata."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model loaded: {filepath}")
    print(f"   Name: {model_data['name']}, Score: {model_data['score']:.4f}")
    return model_data


def get_best_saved_model(model_name=None, base_path="models"):
    """
    Get the best saved model (by score).
    If model_name specified, filter by that model type.
    """
    if not os.path.exists(base_path):
        return None
    
    best_model_data = None
    best_score = -1
    
    for filename in os.listdir(base_path):
        if not filename.endswith('.pkl'):
            continue
        
        if model_name and not filename.startswith(model_name):
            continue
        
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            if model_data['score'] > best_score:
                best_score = model_data['score']
                best_model_data = model_data
                best_model_data['filepath'] = filepath
        except:
            continue
    
    return best_model_data


def list_saved_models(base_path="models"):
    """List all saved models with their scores."""
    if not os.path.exists(base_path):
        print("No saved models found.")
        return []
    
    models = []
    for filename in os.listdir(base_path):
        if not filename.endswith('.pkl'):
            continue
        
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            models.append({
                'filename': filename,
                'name': model_data['name'],
                'score': model_data['score'],
                'timestamp': model_data['timestamp']
            })
        except:
            continue
    
    # Sort by score descending
    models.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nSaved Models:")
    print("-" * 60)
    for m in models:
        print(f"  {m['name']:10} | Score: {m['score']:.4f} | {m['timestamp']} | {m['filename']}")
    print("-" * 60)
    
    return models

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
    """Simple random oversampling (fallback method)."""
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


def smote_oversampling(X, y, k_neighbors=5, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique).
    
    SMOTE creates synthetic samples by interpolating between existing
    minority class samples, which is more effective than simple duplication.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray  
        Target labels.
    k_neighbors : int
        Number of neighbors to use for synthetic sample generation.
    random_state : int
        Random seed for reproducibility.
        
    Returns:
    --------
    X_resampled, y_resampled : Balanced dataset.
    """
    smote = SMOTEScratch(k_neighbors=k_neighbors, random_state=random_state)
    return smote.fit_resample(X, y)

def k_fold_cross_validation(model_class, params, X, y, k=5, use_smote=True):
    """
    K-Fold Cross Validation with optional SMOTE oversampling.
    
    Parameters:
    -----------
    model_class : class
        Model class to instantiate.
    params : dict
        Model hyperparameters.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    k : int
        Number of folds.
    use_smote : bool
        If True, use SMOTE; otherwise use random oversampling.
        
    Returns:
    --------
    float : Mean accuracy across folds.
    """
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
        if use_smote:
            X_fold_train_bal, y_fold_train_bal = smote_oversampling(
                X_fold_train_raw, y_fold_train_raw, k_neighbors=5, random_state=42+i
            )
        else:
            X_fold_train_bal, y_fold_train_bal = random_oversampling(
                X_fold_train_raw, y_fold_train_raw
            )
        
        # 3. TRAIN (On Balanced Data)
        model = model_class(**params)
        
        # Special handling for Decision Tree with post-pruning
        if model_class == DecisionTreeScratch:
            # Use a small portion of training data for pruning validation
            n_prune = int(0.15 * len(X_fold_train_bal))
            prune_idx = np.random.choice(len(X_fold_train_bal), n_prune, replace=False)
            train_mask = np.ones(len(X_fold_train_bal), dtype=bool)
            train_mask[prune_idx] = False
            
            X_prune = X_fold_train_bal[prune_idx]
            y_prune = y_fold_train_bal[prune_idx]
            X_train_final = X_fold_train_bal[train_mask]
            y_train_final = y_fold_train_bal[train_mask]
            
            model.fit(X_train_final, y_train_final, X_val=X_prune, y_val=y_prune)
        else:
            model.fit(X_fold_train_bal, y_fold_train_bal)
        
        # 4. VALIDATE (On Original, Imbalanced Validation Data)
        preds = model.predict(X_fold_val)
        
        # Calculate Accuracy
        acc = np.mean(preds == y_fold_val)
        accuracies.append(acc)
        
    return np.mean(accuracies)

def grid_search(model_class, param_grid, X, y, k=5, n_jobs=2):
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Parallel Grid Search: Testing {len(combinations)} combos on 2+ cores...")
    
    # Define a helper function to run ONE combination
    def run_one_combo(combo):
        params = dict(zip(keys, combo))
        try:
            score = k_fold_cross_validation(model_class, params, X, y, k=k)
            return (params, score)
        except Exception as e:
            return (params, -1)

    # Run in parallel using all available cores (-1)
    results = Parallel(n_jobs, verbose=10)(
        delayed(run_one_combo)(c) for c in combinations
    )
    
    # Find best
    best_params, best_score = max(results, key=lambda x: x[1])
    
    print(f"Best Params: {best_params} | Best CV Score: {best_score:.4f}")
    return best_params, best_score

def main(quick=False, feature_select=True, n_jobs=2, stratified=True):
    print("Checking project structure...")
    
    # 1. Verify Preprocessing
    print("\n[1] Preprocessing Module: OK")
    run_pipeline()

    # 2. Verify Models
    dt = DecisionTreeScratch()
    svm = SVMScratch()
    lr = LogisticRegression()
    ova = OneVsAll(lr)
    
    print(f"[2] Models Loaded Successfully:")
    print(f"    - {dt.__class__.__name__}")
    print(f"    - {svm.__class__.__name__}")
    print(f"    - {lr.__class__.__name__}")
    print(f"    - {ova.__class__.__name__}")

    print("\nProject scaffolding is ready.")

    # ==========================================
    
    # 1. Load Data
    X_train, y_train, X_test_kaggle, test_ids = load_processed_data()
    
    # 3. Handle Imbalance using SMOTE
    print(f"\n[3] Original Class Distribution: {np.unique(y_train, return_counts=True)}")
    X_train_bal, y_train_bal = smote_oversampling(X_train, y_train, k_neighbors=5, random_state=42)
    print(f"[4] SMOTE Balanced Class Distribution: {np.unique(y_train_bal, return_counts=True)}")
    
    # 4. Define Grids for each model
    if quick:
        print("\nQUICK MODE - Using reduced parameter grids")
        # Quick grids for fast testing
        dtl_grid = {
            'max_depth': [None],
            'min_samples_split': [20],
            'min_samples_leaf': [3],
            'criterion': ['entropy'],
            'min_impurity_decrease': [0.002],
            'class_weight': ['balanced'],
        }
        svm_grid = {
            'learning_rate': [0.01],
            'lambda_param': [0.01],
            'n_iters': [2000],
            'batch_size': [32],
            'lr_decay': [True],
            'early_stopping': [True],
            'patience': [30]
        }
        lr_grid = {
            'model_class': [LogisticRegression],
            'learning_rate': [0.01],
            'n_iterations': [500],
            'batch_size': [32],
            'lambda_reg': [0.01],
            'decay_rate': [0.0],
        }
    else:
        # FULL OPTIMIZED GRIDS - targeting 0.8+ accuracy
    
        # --- Model A: Decision Tree (Enhanced with new options) ---
        # Focused grid on most impactful parameters
        dtl_grid = {
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 3, 5],
            'criterion': ['gini', 'entropy'],  # Try both criteria
            'min_impurity_decrease': [0.0, 0.002],  # Minimum gain to split
            'class_weight': ['balanced'],  # Handle imbalance (key for this dataset)
        }
        
        # --- Model B: SVM (with new optimizations) ---
        svm_grid = {
            'learning_rate': [0.01, 0.001],
            'lambda_param': [0.001, 0.01, 0.1],
            'n_iters': [500, 1000, 2000],
            'batch_size': [32, 64],
            'lr_decay': [True],         # Enable learning rate decay
            'early_stopping': [True],    # Enable early stopping
            'patience': [30, 50]         # Patience for early stopping
        }

        # --- Model C: LogReg OvA (with class weights) ---
        lr_grid = {
            'model_class': [LogisticRegression],
            'learning_rate': [0.01, 0.1],
            'n_iterations': [500, 1000],
            'batch_size': [32, 64],
            'lambda_reg': [0.001, 0.01, 0.1],  # L2 regularization strength
            'decay_rate': [0.0, 0.001],  # Learning rate decay
        }

    # 4. Run Grid Search
    print("[4] Tuning")
    print("\n--- Tuning Decision Tree ---")
    best_dtl_params, best_dtl_score = grid_search(DecisionTreeScratch, dtl_grid, X_train, y_train, n_jobs=n_jobs)

    print("\n--- Tuning SVM (OvA) ---")
    best_svm_params, best_svm_score = grid_search(SVMScratch, svm_grid, X_train, y_train, n_jobs=n_jobs)

    print("\n--- Tuning LogReg (OvA) ---")
    best_lr_params, best_lr_score = grid_search(OneVsAll, lr_grid, X_train, y_train, n_jobs=n_jobs)

    # 5. Final Training & Kaggle Submission
    # We select the best model based on CV score
    scores = {
        "DTL": best_dtl_score,
        "LogReg": best_lr_score,
        "SVM": best_svm_score
    }
    
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:10}: {score:.4f}")
    print("=" * 50)
    
    winner_name = max(scores, key=scores.get)
    print(f"\nWinning Model: {winner_name} (Acc: {scores[winner_name]:.4f})")
    
    # Generate timestamp for file naming
    date_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    # Train and save ALL models (for flexibility)
    print("\n[5] Training and saving all models...")
    
    # --- Train DTL ---
    dtl_model = DecisionTreeScratch(**best_dtl_params)
    n_prune = int(0.15 * len(X_train_bal))
    prune_idx = np.random.choice(len(X_train_bal), n_prune, replace=False)
    train_mask = np.ones(len(X_train_bal), dtype=bool)
    train_mask[prune_idx] = False
    dtl_model.fit(X_train_bal[train_mask], y_train_bal[train_mask], 
                  X_val=X_train_bal[prune_idx], y_val=y_train_bal[prune_idx])
    save_model(dtl_model, "DTL", best_dtl_params, best_dtl_score)
    
    # --- Train SVM ---
    svm_model = SVMScratch(**best_svm_params)
    svm_model.fit(X_train_bal, y_train_bal)
    save_model(svm_model, "SVM", best_svm_params, best_svm_score)
    
    # --- Train LogReg ---
    lr_model = OneVsAll(**best_lr_params)
    lr_model.fit(X_train_bal, y_train_bal)
    save_model(lr_model, "LogReg", best_lr_params, best_lr_score)
    
    # Select winner for submission
    if winner_name == "DTL":
        final_model = dtl_model
        print(f"\nDecision Tree Info:")
        print(f"  Depth: {final_model.get_depth()}")
        print(f"  Leaves: {final_model.get_n_leaves()}")
    elif winner_name == "SVM":
        final_model = svm_model
    else:
        final_model = lr_model
    
    # [BONUS] If SVM wins, save training visualization
    if winner_name == "SVM":
        print("\n[BONUS] Generating SVM Training Visualization...")
        plot_path = f"images/svm_training_progress_{date_str}.png"
        final_model.plot_training_history(save_path=plot_path, show=False)
        
        summary = final_model.get_training_summary()
        print("\nSVM Training Summary:")
        class_names = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        for cls, stats in summary.items():
            print(f"  {class_names.get(cls, cls)}: Epochs={stats['epochs_trained']}, "
                  f"Final Acc={stats['final_accuracy']:.4f}, Best Acc={stats['best_accuracy']:.4f}")
    
    # Predict on Kaggle Test Set
    numeric_preds = final_model.predict(X_test_kaggle)

    reverse_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    
    string_preds = [reverse_map[pred] for pred in numeric_preds]
    
    # Save CSV
    submission_df = pd.DataFrame({
        'Student_ID': test_ids,
        'Target': string_preds 
    })

    submission_path = f"data/submit/submission_{winner_name}_{date_str}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    
    # Show all saved models
    list_saved_models()


def predict_only(model_path=None, model_name=None):
    """
    Load a saved model and generate predictions without training.
    
    Usage:
        python main.py --predict                    # Use best saved model
        python main.py --predict --model LogReg    # Use best LogReg model
        python main.py --predict --model-path models/LogReg_xxx.pkl
    """
    print("=" * 60)
    print("PREDICT-ONLY MODE (Loading saved model)")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test_kaggle, test_ids = load_processed_data()
    
    # Load model
    if model_path:
        model_data = load_model(model_path)
    else:
        model_data = get_best_saved_model(model_name)
        if model_data is None:
            print("No saved models found. Run training first.")
            return
    
    final_model = model_data['model']
    model_name = model_data['name']
    
    # Apply feature selection if model was trained with it
    selected_features = model_data.get('selected_features')
    if selected_features is not None:
        print(f"  Applying saved feature selection ({len(selected_features)} features)")
        X_test_kaggle = X_test_kaggle[:, selected_features]
    
    # Predict
    numeric_preds = final_model.predict(X_test_kaggle)
    
    reverse_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    string_preds = [reverse_map[pred] for pred in numeric_preds]
    
    # Save
    date_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    submission_df = pd.DataFrame({
        'Student_ID': test_ids,
        'Target': string_preds 
    })
    
    submission_path = f"data/submit/submission_{model_name}_{date_str}.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Model Training/Prediction')
    parser.add_argument('-p', '--predict', action='store_true', 
                        help='Predict-only mode (load saved model)')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Model name to load (DTL, SVM, LogReg)')
    parser.add_argument('-P', '--model-path', type=str, default=None,
                        help='Path to specific model file')
    parser.add_argument('-l', '--list-models', action='store_true',
                        help='List all saved models')
    parser.add_argument('-q', '--quick', action='store_true',
                        help='Quick mode with reduced hyperparameter grid')
    parser.add_argument('-j', '--jobs', type=int, default=2,
                        help='Number of job(s) used in training')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_saved_models()
    elif args.predict:
        predict_only(model_path=args.model_path, model_name=args.model)
    else:
        main(quick=args.quick, n_jobs=args.jobs)
