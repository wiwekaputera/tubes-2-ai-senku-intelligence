import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist

class SMOTEScratch:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) - From Scratch.
    
    SMOTE generates synthetic samples by interpolating between existing 
    minority class samples and their k-nearest neighbors.
    
    This is DATA AUGMENTATION, not a model - compliant with project rules.
    """
    
    def __init__(self, k_neighbors=5, random_state=42):
        """
        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors to use for synthetic sample generation.
        random_state : int
            Random seed for reproducibility.
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    
    def _get_neighbors(self, X, sample_idx, k):
        """
        Find k-nearest neighbors for a sample using Euclidean distance.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix for a single class.
        sample_idx : int
            Index of the sample to find neighbors for.
        k : int
            Number of neighbors to find.
            
        Returns:
        --------
        np.ndarray : Indices of k-nearest neighbors.
        """
        # Calculate distances from this sample to all others
        sample = X[sample_idx].reshape(1, -1)
        distances = cdist(sample, X, metric='euclidean').flatten()
        
        # Set distance to self as infinity so it's not selected
        distances[sample_idx] = np.inf
        
        # Get indices of k smallest distances
        neighbor_indices = np.argsort(distances)[:k]
        return neighbor_indices
    
    def _generate_synthetic_sample(self, sample, neighbor, rng):
        """
        Generate a synthetic sample between sample and its neighbor.
        
        synthetic = sample + rand(0,1) * (neighbor - sample)
        
        Parameters:
        -----------
        sample : np.ndarray
            The original sample.
        neighbor : np.ndarray
            A neighbor of the sample.
        rng : np.random.Generator
            Random number generator.
            
        Returns:
        --------
        np.ndarray : Synthetic sample.
        """
        # Random interpolation factor between 0 and 1
        alpha = rng.random()
        
        # Linear interpolation
        synthetic = sample + alpha * (neighbor - sample)
        return synthetic
    
    def fit_resample(self, X, y):
        """
        Resample the dataset to balance classes using SMOTE.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Target labels (n_samples,).
            
        Returns:
        --------
        X_resampled : np.ndarray
            Resampled feature matrix.
        y_resampled : np.ndarray
            Resampled target labels.
        """
        rng = np.random.default_rng(self.random_state)
        
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]
        
        for cls, count in zip(classes, counts):
            if count >= max_count:
                continue  # Skip majority class
            
            # Get samples of this class
            cls_indices = np.where(y == cls)[0]
            X_cls = X[cls_indices]
            
            # Number of synthetic samples needed
            n_synthetic = max_count - count
            
            # Adjust k if class has fewer samples than k_neighbors
            k = min(self.k_neighbors, len(X_cls) - 1)
            if k < 1:
                # Fallback to random oversampling if class is too small
                oversample_indices = rng.choice(len(X_cls), size=n_synthetic, replace=True)
                X_resampled.append(X_cls[oversample_indices])
                y_resampled.append(np.full(n_synthetic, cls))
                continue
            
            # Generate synthetic samples
            synthetic_samples = []
            
            for _ in range(n_synthetic):
                # Randomly pick a sample from minority class
                sample_idx = rng.integers(0, len(X_cls))
                sample = X_cls[sample_idx]
                
                # Find its k-nearest neighbors
                neighbor_indices = self._get_neighbors(X_cls, sample_idx, k)
                
                # Randomly pick one neighbor
                neighbor_idx = rng.choice(neighbor_indices)
                neighbor = X_cls[neighbor_idx]
                
                # Generate synthetic sample
                synthetic = self._generate_synthetic_sample(sample, neighbor, rng)
                synthetic_samples.append(synthetic)
            
            X_resampled.append(np.array(synthetic_samples))
            y_resampled.append(np.full(n_synthetic, cls))
        
        # Concatenate all
        X_final = np.vstack(X_resampled)
        y_final = np.hstack(y_resampled)
        
        # Shuffle
        shuffle_idx = rng.permutation(len(X_final))
        
        return X_final[shuffle_idx], y_final[shuffle_idx]


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
    if "Student_ID" in test_df.columns:
        id_col = "Student_ID"

    if id_col:
        print(f"ID column found: '{id_col}'. Extracting and dropping.")
        test_ids = test_df[id_col].values
        test_df = test_df.drop(columns=[id_col])

        # Save IDs
        np.save(os.path.join(processed_path, "test_ids.npy"), test_ids)

        # Drop from train
        if id_col in train_df.columns:
            train_df = train_df.drop(columns=[id_col])
    else:
        print("WARNING: No ID column found.")

    return train_df, test_df


def add_derived_features(df):
    """
    Create new features based on domain knowledge (Interaction Features).
    Enhanced with more predictive features for student dropout prediction.
    """
    # 1. Grade Trend (2nd Sem - 1st Sem) - Positive = improving
    df["Grade_Trend"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]

    # 2. Approval Trend (2nd Sem - 1st Sem)
    df["Approval_Trend"] = df["Curricular units 2nd sem (approved)"] - df["Curricular units 1st sem (approved)"]

    # 3. Global Approval Ratio (Approved / Enrolled)
    total_enrolled = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
    total_approved = df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
    df["Approval_Ratio"] = np.where(total_enrolled > 0, total_approved / total_enrolled, 0.0)

    # 4. Admission Gap (1st Sem Grade - Scaled Admission Grade)
    df["Admission_Gap"] = df["Curricular units 1st sem (grade)"] - (df["Admission grade"] / 10.0)

    # 5. Pressure Indicator (Evaluations / Enrolled)
    total_evaluations = df["Curricular units 1st sem (evaluations)"] + df["Curricular units 2nd sem (evaluations)"]
    df["Pressure_Indicator"] = np.where(
        total_enrolled > 0, 
        total_evaluations / total_enrolled, 
        0.0
    )

    # 6. Total Failed Credits
    failed_sem1 = df["Curricular units 1st sem (enrolled)"] - df["Curricular units 1st sem (approved)"]
    failed_sem2 = df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 2nd sem (approved)"]
    df["Total_Failed_Credits"] = failed_sem1 + failed_sem2

    # 7. Failure Rates per Semester
    df["Sem1_Failure_Rate"] = np.where(
        df["Curricular units 1st sem (enrolled)"] > 0,
        failed_sem1 / df["Curricular units 1st sem (enrolled)"],
        0.0
    )
    df["Sem2_Failure_Rate"] = np.where(
        df["Curricular units 2nd sem (enrolled)"] > 0,
        failed_sem2 / df["Curricular units 2nd sem (enrolled)"],
        0.0
    )

    # 8. Average Grade (weighted by credits)
    total_grades = df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]
    df["Avg_Grade"] = np.where(total_approved > 0, total_grades / 2, 0.0)
    
    # 9. Credit Efficiency - How many credits approved vs evaluated
    df["Credit_Efficiency"] = np.where(
        total_evaluations > 0,
        total_approved / total_evaluations,
        0.0
    )
    
    # 10. Academic Engagement Score (enrolled + evaluations weighted)
    df["Academic_Engagement"] = total_enrolled + 0.5 * total_evaluations
    
    # 11. Zero Performance Flag - Student with 0 approved in any semester
    df["Zero_Sem1_Approved"] = (df["Curricular units 1st sem (approved)"] == 0).astype(int)
    df["Zero_Sem2_Approved"] = (df["Curricular units 2nd sem (approved)"] == 0).astype(int)
    
    # 12. Grade Consistency (lower std = more consistent)
    # Using absolute difference as a proxy for std with 2 points
    df["Grade_Consistency"] = np.abs(
        df["Curricular units 1st sem (grade)"] - df["Curricular units 2nd sem (grade)"]
    )
    
    # 13. Economic Stress Index (Unemployment + Inflation)
    df["Econ_Stress_Index"] = df["Unemployment rate"] + df["Inflation rate"]

    # 14. Financial Support Gap (Scholarship - Debtor)
    df["Financial_Support_Gap"] = df["Scholarship holder"] - df["Debtor"]
    
    # 15. Financial Risk Score (Debtor + No Scholarship + Tuition not up to date)
    df["Financial_Risk"] = df["Debtor"] + (1 - df["Scholarship holder"]) + (1 - df["Tuition fees up to date"])
    
    # 16. GDP per capita stress (inverse relationship with success)
    # Higher GDP growth might reduce dropout
    df["GDP_Stress"] = -df["GDP"]  # Negative because higher GDP = less stress
    
    # 17. Age at Enrollment (derived if not present, using a proxy)
    df["Age_Group"] = (df["Age at enrollment"] > 23).astype(int)  # Mature student flag
    
    # 18. Parent Education Gap
    df["Parent_Edu_Gap"] = np.abs(
        df["Mother's qualification"].astype(float) - df["Father's qualification"].astype(float)
    )
    
    # 19. Parent Education Average (higher = more support)
    df["Parent_Edu_Avg"] = (
        df["Mother's qualification"].astype(float) + df["Father's qualification"].astype(float)
    ) / 2
    
    # 20. First Generation Risk - both parents low qualification
    # Assuming qualification codes: lower numbers = lower education
    low_edu_threshold = 10  # Adjust based on actual encoding
    df["First_Gen_Risk"] = (
        (df["Mother's qualification"].astype(float) < low_edu_threshold) & 
        (df["Father's qualification"].astype(float) < low_edu_threshold)
    ).astype(int)
    
    # 21. Scholarship * Grade interaction (scholarship impact on performance)
    df["Scholarship_Grade_Interaction"] = df["Scholarship holder"] * df["Avg_Grade"]
    
    # 22. Debtor * Failure interaction (financial stress impact)
    df["Debtor_Failure_Interaction"] = df["Debtor"] * (df["Sem1_Failure_Rate"] + df["Sem2_Failure_Rate"])
    
    # 23. Age * Performance interaction
    df["Age_Performance_Interaction"] = df["Age at enrollment"] * df["Approval_Ratio"]

    print(f"Derived features added: {23} new features including academic, socioeconomic, demographic, and interaction features.")
    return df


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


def select_features(X_train, X_val, X_test, correlation_threshold=0.95):
    """
    Feature Selection:
    1. Drop Constant Columns.
    2. Drop Highly Correlated Columns (> threshold).
    """
    print(f"\n=== Feature Selection (Threshold: {correlation_threshold}) ===")
    initial_count = X_train.shape[1]

    # 1. Drop Constant Columns
    std = np.std(X_train, axis=0)
    constant_indices = np.where(std == 0)[0]
    
    if len(constant_indices) > 0:
        print(f"Dropping {len(constant_indices)} constant features.")
        X_train = np.delete(X_train, constant_indices, axis=1)
        X_val = np.delete(X_val, constant_indices, axis=1)
        X_test = np.delete(X_test, constant_indices, axis=1)
    
    # 2. Drop Highly Correlated Features
    df_corr = pd.DataFrame(X_train)
    corr_matrix = df_corr.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features (> {correlation_threshold}).")
        # Convert column names (which are just ints 0..N) back to indices
        to_drop_indices = to_drop # They match because we made a fresh DF
        
        X_train = np.delete(X_train, to_drop_indices, axis=1)
        X_val = np.delete(X_val, to_drop_indices, axis=1)
        X_test = np.delete(X_test, to_drop_indices, axis=1)

    final_count = X_train.shape[1]
    print(f"Features reduced from {initial_count} to {final_count}.")
    
    return X_train, X_val, X_test


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
    2. One-Hot Encodes Categorical Features based on explicit domain knowledge.
    """
    # 1. Encode Target
    target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    if "Target" not in train_df.columns:
        raise ValueError("Target column not found")

    y_train = train_df["Target"].map(target_map).values

    # Drop Target from features
    X_train_raw = train_df.drop(columns=["Target"])
    X_test_raw = test_df.copy()  # Test set has no target

    # 2. Define Explicit Feature Types
    # Nominal: No intrinsic order -> One-Hot Encode
    nominal_cols = [
        "Marital status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation"
    ]

    # Binary/Ordinal/Numeric: Keep as is (Ordinal is treated as numeric here for simplicity)
    # Note: "Application order" is ordinal (1-8), kept numeric.
    # Note: Economic indicators (GDP, etc) are numeric.
    
    # Verify columns exist
    present_nominal = [c for c in nominal_cols if c in X_train_raw.columns]
    
    print(f"🔠 One-Hot Encoding Cols ({len(present_nominal)}): {present_nominal}")

    # 3. One-Hot Encoding
    # We combine train/test temporarily to ensure same columns exist in both
    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    
    # Explicitly cast nominal cols to object to force get_dummies to treat them as cat
    for col in present_nominal:
        combined[col] = combined[col].astype(str)

    combined_encoded = pd.get_dummies(combined, columns=present_nominal, dtype=int, drop_first=True)

    # Split back
    X_train_encoded = combined_encoded.iloc[: len(X_train_raw)]
    X_test_encoded = combined_encoded.iloc[len(X_train_raw) :]

    print(f"Features encoded. New feature count: {X_train_encoded.shape[1]}")

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


def run_full_pipeline(
    raw_path="data/raw", processed_path="data/processed", handle_outliers_flag=True
):
    print("=" * 60 + "\nSTARTING REFACTORED PIPELINE (One-Hot + CPU)\n" + "=" * 60)

    train_df, test_df = load_data(raw_path)
    train_df = remove_duplicates(train_df)
    train_df, test_df = extract_and_drop_ids(train_df, test_df, processed_path)

    # FEATURE ENGINEERING
    train_df = add_derived_features(train_df)
    test_df = add_derived_features(test_df)

    # ENCODING STEP
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
    
    # FEATURE SELECTION (New Step)
    X_train, X_val, X_test = select_features(X_train, X_val, X_test, correlation_threshold=0.95)

    save_processed_data(
        X_train, y_train, X_val, y_val, X_test, mean, std, processed_path
    )

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
