import pandas as pd
import numpy as np
import os

# UTILITY FUNCTIONS
def check_data_types(df):
    """Check for non-numeric columns that need encoding."""
    print("\n=== Data Types Check ===")
    print(df.dtypes)
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"Non-numeric columns found: {non_numeric}")
    else:
        print("All columns are numeric.")
    return non_numeric


# INDIVIDUAL PREPROCESSING FUNCTIONS
def load_data(raw_path="data/raw"):
    """Load train and test CSVs."""
    train_df = pd.read_csv(os.path.join(raw_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(raw_path, "test.csv"))
    print(f"Loaded train shape: {train_df.shape}")
    print(f"Loaded test shape: {test_df.shape}")
    return train_df, test_df


def remove_duplicates(train_df):
    """Remove duplicate rows from training data."""
    initial = train_df.shape[0]
    train_df = train_df.drop_duplicates()
    removed = initial - train_df.shape[0]
    print(f"Removed {removed} duplicate rows.")
    return train_df


def extract_and_drop_ids(train_df, test_df, processed_path="data/processed"):
    """
    Extract Student_ID from test set, save it, and drop from both datasets.
    """
    os.makedirs(processed_path, exist_ok=True)

    # Find ID column
    id_col = None
    if "id" in test_df.columns:
        id_col = "id"
    elif "Student_ID" in test_df.columns:
        id_col = "Student_ID"

    if id_col:
        print(f"ID column found: '{id_col}'. Extracting and dropping.")
        test_ids = test_df[id_col].values
        test_df = test_df.drop(columns=[id_col])

        # Save IDs
        np.save(os.path.join(processed_path, "test_ids.npy"), test_ids)

        # Drop from train if present
        if id_col in train_df.columns:
            train_df = train_df.drop(columns=[id_col])
    else:
        print("WARNING: No ID column found.")

    return train_df, test_df


def encode_target(train_df):
    """
    Encode Target column: Dropout->0, Enrolled->1, Graduate->2
    Returns X (features) and y (encoded labels)
    """
    target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}

    if "Target" not in train_df.columns:
        raise ValueError("Target column not found in train.csv")

    y = train_df["Target"].map(target_map).values
    X = train_df.drop(columns=["Target"])

    print(f"Target encoded. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def split_train_val(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and validation sets (80/20).
    """
    indices = np.arange(len(y))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    split_idx = int((1 - test_size) * len(y))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train = X.iloc[train_idx].values
    y_train = y[train_idx]
    X_val = X.iloc[val_idx].values
    y_val = y[val_idx]

    print(f"Split complete. Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def impute_missing_values(X_train, X_val, X_test):
    """
    Impute missing values using mean (fit on train only).
    """
    col_means = np.nanmean(X_train, axis=0)

    # Apply to train
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_means, inds[1])

    # Apply to val
    inds = np.where(np.isnan(X_val))
    X_val[inds] = np.take(col_means, inds[1])

    # Apply to test
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_means, inds[1])

    print("Missing values imputed using mean.")
    return X_train, X_val, X_test


def handle_outliers(X_train, X_val, X_test, method="clip", percentile=1):
    """
    Handle outliers using clipping.
    Clips values at the specified percentile based on training data.
    """
    if method == "clip":
        lower = np.percentile(X_train, percentile, axis=0)
        upper = np.percentile(X_train, 100 - percentile, axis=0)

        X_train = np.clip(X_train, lower, upper)
        X_val = np.clip(X_val, lower, upper)
        X_test = np.clip(X_test, lower, upper)

        print(f"Outliers clipped at {percentile}th and {100-percentile}th percentile.")

    return X_train, X_val, X_test


def standardize_features(X_train, X_val, X_test):
    """
    Standardize features using Z-score (fit on train only).
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0  # Avoid division by zero

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print("Features standardized using Z-score.")
    return X_train, X_val, X_test, mean, std


def save_processed_data(
    X_train, y_train, X_val, y_val, X_test, mean, std, processed_path="data/processed"
):
    """
    Save all processed arrays to .npy files.
    """
    os.makedirs(processed_path, exist_ok=True)

    np.save(os.path.join(processed_path, "X_train.npy"), X_train)
    np.save(os.path.join(processed_path, "y_train.npy"), y_train)
    np.save(os.path.join(processed_path, "X_val.npy"), X_val)
    np.save(os.path.join(processed_path, "y_val.npy"), y_val)
    np.save(os.path.join(processed_path, "X_test_kaggle.npy"), X_test)
    np.save(os.path.join(processed_path, "scaler_mean.npy"), mean)
    np.save(os.path.join(processed_path, "scaler_std.npy"), std)

    print(f"All files saved to {processed_path}/")


def encode_data(train_df, test_df):
    """
    1. Encodes Target (Dropout -> 0)
    2. One-Hot Encodes Categorical Features (Marital, Course, etc.)
    """
    # 1. Encode Target
    target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    if "Target" not in train_df.columns:
        raise ValueError("Target column not found")
    
    y_train = train_df["Target"].map(target_map).values
    
    # Drop Target from features
    X_train_raw = train_df.drop(columns=["Target"])
    X_test_raw = test_df.copy() # Test set has no target
    
    # 2. Identify Categorical Columns (Those with few unique values)
    # In this dataset, many "int" columns are actually categories (Course, Marital, etc.)
    # Rule of thumb: < 20 unique values = Categorical
    cat_cols = [col for col in X_train_raw.columns if X_train_raw[col].nunique() < 20]
    num_cols = [col for col in X_train_raw.columns if col not in cat_cols]
    
    print(f"🔠 Categorical Cols ({len(cat_cols)}): {cat_cols}")
    print(f"🔢 Numerical Cols ({len(num_cols)}): {num_cols}")
    
    # 3. One-Hot Encoding (using Pandas get_dummies)
    # We combine train/test temporarily to ensure same columns exist in both
    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, dtype=int)
    
    # Split back
    X_train_encoded = combined_encoded.iloc[:len(X_train_raw)]
    X_test_encoded = combined_encoded.iloc[len(X_train_raw):]
    
    return X_train_encoded, y_train, X_test_encoded

def split_train_val(X, y, test_size=0.2, random_state=42):
    # Standard shuffle split
    indices = np.arange(len(y))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    split_idx = int((1 - test_size) * len(y))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    # X is a DataFrame from encode_data
    return X.iloc[train_idx].values, X.iloc[val_idx].values, y[train_idx], y[val_idx]

def run_full_pipeline(raw_path="data/raw", processed_path="data/processed", handle_outliers_flag=True):
    print("="*60 + "\nSTARTING REFACTORED PIPELINE (One-Hot + CPU)\n" + "="*60)
    
    train_df, test_df = load_data(raw_path)
    train_df = remove_duplicates(train_df)
    train_df, test_df = extract_and_drop_ids(train_df, test_df, processed_path)
    
    # --- NEW ENCODING STEP ---
    X_train_df, y_train_all, X_test_df = encode_data(train_df, test_df)
    
    # Split
    X_train, X_val, y_train, y_val = split_train_val(X_train_df, y_train_all)
    X_test = X_test_df.values
    
    # Impute & Outliers & Standardize (Same as before)
    # Note: Standardizing One-Hot columns is debatable, but acceptable for SVM.
    X_train, X_val, X_test = impute_missing_values(X_train, X_val, X_test)
    
    if handle_outliers_flag:
        # Only outlier clipping on the continuous columns might be better, 
        # but global clipping is "safe enough" for this assignment level.
        X_train, X_val, X_test = handle_outliers(X_train, X_val, X_test)
        
    X_train, X_val, X_test, mean, std = standardize_features(X_train, X_val, X_test)
    
    save_processed_data(X_train, y_train, X_val, y_val, X_test, mean, std, processed_path)

    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f" X_train: {X_train.shape}")
    print(f" y_train: {y_train.shape}")
    print(f" X_val: {X_val.shape}")
    print(f" y_val: {y_val.shape}")
    print(f" X_test: {X_test.shape}")
    print(f" All files saved to '{processed_path}/'")


if __name__ == "__main__":
    # Run the full pipeline
    run_full_pipeline()
