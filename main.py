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

def save_model(model, model_name, params, score, selected_features=None, base_path="models"):
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
        'selected_features': selected_features,  # Store feature selection
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

def stratified_k_fold_split(X, y, k=5, random_state=42):
    """
    Generate stratified k-fold indices.
    Ensures each fold has approximately the same class distribution.
    """
    np.random.seed(random_state)
    
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}
    
    # Shuffle indices within each class
    for c in classes:
        np.random.shuffle(class_indices[c])
    
    # Create folds
    folds = [[] for _ in range(k)]
    
    for c in classes:
        indices = class_indices[c]
        n = len(indices)
        fold_sizes = [(n // k) + (1 if i < (n % k) else 0) for i in range(k)]
        
        current = 0
        for fold_idx, size in enumerate(fold_sizes):
            folds[fold_idx].extend(indices[current:current + size])
            current += size
    
    # Convert to arrays and shuffle within each fold
    for i in range(k):
        folds[i] = np.array(folds[i])
        np.random.shuffle(folds[i])
    
    return folds


def select_features_by_correlation(X, y, threshold=0.02):
    """
    Select features based on correlation with target.
    Returns indices of features with correlation > threshold.
    """
    correlations = []
    for i in range(X.shape[1]):
        # Point-biserial correlation approximation
        corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        if np.isnan(corr):
            corr = 0
        correlations.append(corr)
    
    correlations = np.array(correlations)
    selected = np.where(correlations >= threshold)[0]
    
    print(f"  Feature Selection: {len(selected)}/{X.shape[1]} features kept (corr >= {threshold})")
    return selected


def compute_f_statistic(X, y):
    """
    Compute F-statistic for feature selection (ANOVA F-test).
    Higher F = better separation between classes.
    """
    classes = np.unique(y)
    n_samples, n_features = X.shape
    overall_mean = np.mean(X, axis=0)
    
    f_stats = np.zeros(n_features)
    
    for j in range(n_features):
        # Between-group variance
        ss_between = 0
        ss_within = 0
        
        for c in classes:
            mask = y == c
            group = X[mask, j]
            group_mean = np.mean(group)
            n_g = len(group)
            
            ss_between += n_g * (group_mean - overall_mean[j]) ** 2
            ss_within += np.sum((group - group_mean) ** 2)
        
        # F-statistic
        df_between = len(classes) - 1
        df_within = n_samples - len(classes)
        
        if ss_within == 0 or df_within == 0:
            f_stats[j] = 0
        else:
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            f_stats[j] = ms_between / (ms_within + 1e-10)
    
    return f_stats


def select_features_by_f_test(X, y, k=80):
    """
    Select top-k features by F-statistic (ANOVA).
    """
    f_stats = compute_f_statistic(X, y)
    top_k_idx = np.argsort(f_stats)[::-1][:k]
    print(f"  Feature Selection: Top {k}/{X.shape[1]} features by F-statistic")
    return np.sort(top_k_idx)


def k_fold_cross_validation(model_class, params, X, y, k=5, use_smote=True, stratified=True):
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
    stratified : bool
        If True, use stratified splitting (maintains class distribution).
        
    Returns:
    --------
    float : Mean accuracy across folds.
    """
    # Stratified or random split
    if stratified:
        folds = stratified_k_fold_split(X, y, k=k)
    else:
        fold_size = len(X) // k
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]
    
    accuracies = []
    
    for i in range(k):
        # 1. SPLIT (Raw, Imbalanced Data) using stratified folds
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
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
    """
    Grid search with parallel processing.
    
    Note: n_jobs=2 is often optimal. More cores can be slower due to:
    - Process spawning overhead
    - Memory copying to each worker
    - Memory bandwidth bottlenecks
    """
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Parallel Grid Search: Testing {len(combinations)} combos on {n_jobs} cores...")
    
    # Define a helper function to run ONE combination
    def run_one_combo(combo):
        params = dict(zip(keys, combo))
        try:
            score = k_fold_cross_validation(model_class, params, X, y, k=k)
            return (params, score)
        except Exception as e:
            return (params, -1)

    # Run in parallel (default 2 cores - often faster than using all cores)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_combo)(c) for c in combinations
    )
    
    # Find best
    best_params, best_score = max(results, key=lambda x: x[1])
    
    print(f"Best Params: {best_params} | Best CV Score: {best_score:.4f}")
    return best_params, best_score

def main(quick=False, feature_select=True, n_jobs=2):
    print("Checking project structure...")
    
    # 1. Verify Preprocessing
    print("\n[1] Preprocessing Module: OK")
    run_pipeline()

    # 2. Verify Models
    dt = DecisionTreeScratch()
    svm = SVMScratch()
    lr = LogisticRegression()
    ova = OneVsAll()
    
    print(f"[2] Models Loaded Successfully:")
    print(f"    - {dt.__class__.__name__}")
    print(f"    - {svm.__class__.__name__}")
    print(f"    - {lr.__class__.__name__}")
    print(f"    - {ova.__class__.__name__}")

    print("\nProject scaffolding is ready.")

    # ==========================================
    
    # 1. Load Data
    X_train, y_train, X_test_kaggle, test_ids = load_processed_data()
    
    # 2. FEATURE SELECTION (keep most predictive features)
    if feature_select:
        print("\n[2.5] Feature Selection using F-statistic...")
        selected_features = select_features_by_f_test(X_train, y_train, k=90)
        X_train = X_train[:, selected_features]
        X_test_kaggle = X_test_kaggle[:, selected_features]
        print(f"      New shape: {X_train.shape}")
    else:
        selected_features = None
    
    # 3. Handle Imbalance using SMOTE
    print(f"\n[3] Original Class Distribution: {np.unique(y_train, return_counts=True)}")
    X_train_bal, y_train_bal = smote_oversampling(X_train, y_train, k_neighbors=5, random_state=42)
    print(f"[4] SMOTE Balanced Class Distribution: {np.unique(y_train_bal, return_counts=True)}")
    
    # 4. Define Grids for each model
    if quick:
        print("\nQUICK MODE - Using reduced parameter grids")
        # Quick grids for fast testing
        dtl_grid = {
            'max_depth': [12, 18],
            'min_samples_split': [8],
            'min_samples_leaf': [3],
            'criterion': ['entropy'],
            'min_impurity_decrease': [0.0],
            'class_weight': ['balanced'],
        }
        svm_grid = {
            'learning_rate': [0.01],
            'lambda_param': [0.001],
            'n_iters': [1000],
            'batch_size': [32],
            'lr_decay': [True],
            'early_stopping': [True],
            'patience': [50]
        }
        lr_grid = {
            'model_class': [LogisticRegression],
            'learning_rate': [0.1],
            'n_iterations': [1000],
            'batch_size': [64],
            'lambda_reg': [0.001],
            'decay_rate': [0.001],
            'class_weight': ['balanced'],
            'early_stopping': [True],
            'patience': [100],
        }
    else:
        # FULL OPTIMIZED GRIDS - targeting 0.8+ accuracy
        
        # --- Model A: Decision Tree ---
        dtl_grid = {
            'max_depth': [10, 14, 18, 22],
            'min_samples_split': [6, 10, 15],
            'min_samples_leaf': [2, 4, 6],
            'criterion': ['gini', 'entropy'],
            'min_impurity_decrease': [0.0, 0.0005],
            'class_weight': ['balanced'],
            'max_features': [None, 'sqrt'],  # Add feature randomness
        }
        
        # --- Model B: SVM (with optimizations) ---
        svm_grid = {
            'learning_rate': [0.005, 0.01, 0.02],
            'lambda_param': [0.0005, 0.001, 0.003],
            'n_iters': [1500, 2500],
            'batch_size': [32, 64],
            'lr_decay': [True],
            'early_stopping': [True],
            'patience': [60, 100]
        }

        # --- Model C: LogReg OvA (FINAL PUSH) ---
        lr_grid = {
            'model_class': [LogisticRegression],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'n_iterations': [1000, 1500, 2500],
            'batch_size': [32, 64, 128],
            'lambda_reg': [0.0001, 0.0005, 0.001, 0.005],
            'decay_rate': [0.0005, 0.001],
            'class_weight': ['balanced'],
            'early_stopping': [True],
            'patience': [80, 120],
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
    save_model(dtl_model, "DTL", best_dtl_params, best_dtl_score, selected_features)
    
    # --- Train SVM ---
    svm_model = SVMScratch(**best_svm_params)
    svm_model.fit(X_train_bal, y_train_bal)
    save_model(svm_model, "SVM", best_svm_params, best_svm_score, selected_features)
    
    # --- Train LogReg ---
    lr_model = OneVsAll(**best_lr_params)
    lr_model.fit(X_train_bal, y_train_bal)
    save_model(lr_model, "LogReg", best_lr_params, best_lr_score, selected_features)
    
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
    parser.add_argument('-N', '--no-feature-select', action='store_true',
                        help='Disable F-statistic feature selection')
    parser.add_argument('-j', '--jobs', type=int, default=2,
                        help='Number of job(s) used in training')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_saved_models()
    elif args.predict:
        predict_only(model_path=args.model_path, model_name=args.model)
    else:
        main(quick=args.quick, feature_select=not args.no_feature_select, n_jobs=args.jobs)
