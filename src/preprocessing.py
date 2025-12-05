import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist

class SMOTEScratch:
    """
    SMOTE from scratch untuk bantu oversampling kelas minoritas
    """
    
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    
    # Find index untuk k nearest neighbors
    def _get_neighbors(self, X, sample_idx, k):
        sample = X[sample_idx].reshape(1, -1)
        distances = cdist(sample, X, metric='euclidean').flatten()
        distances[sample_idx] = np.inf
        neighbor_indices = np.argsort(distances)[:k]
        return neighbor_indices
    
    # Generate a point between sample and neighbor
    def _generate_synthetic_sample(self, sample, neighbor, rng):
        alpha = rng.random()
        synthetic = sample + alpha * (neighbor - sample)
        return synthetic
    
    # Balance and resample semua kelasnya
    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        
        classes, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]
        
        for cls, count in zip(classes, counts):
            if count >= max_count:
                continue
            
            cls_indices = np.where(y == cls)[0]
            X_cls = X[cls_indices]
            
            n_synthetic = max_count - count
            
            k = min(self.k_neighbors, len(X_cls) - 1)
            if k < 1:
                oversample_indices = rng.choice(len(X_cls), size=n_synthetic, replace=True)
                X_resampled.append(X_cls[oversample_indices])
                y_resampled.append(np.full(n_synthetic, cls))
                continue
            
            synthetic_samples = []
            
            for _ in range(n_synthetic):
                sample_idx = rng.integers(0, len(X_cls))
                sample = X_cls[sample_idx]
                
                neighbor_indices = self._get_neighbors(X_cls, sample_idx, k)
                neighbor_idx = rng.choice(neighbor_indices)
                neighbor = X_cls[neighbor_idx]
                
                synthetic = self._generate_synthetic_sample(sample, neighbor, rng)
                synthetic_samples.append(synthetic)
            
            X_resampled.append(np.array(synthetic_samples))
            y_resampled.append(np.full(n_synthetic, cls))
        
        X_final = np.vstack(X_resampled)
        y_final = np.hstack(y_resampled)
        
        shuffle_idx = rng.permutation(len(X_final))
        
        return X_final[shuffle_idx], y_final[shuffle_idx]

# Find non numeric columns
def check_data_types(df):
    print("\n=== Data Types Check ===")
    print(df.dtypes)
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"Non-numeric columns found: {non_numeric}")
    else:
        print("All columns are numeric.")
    return non_numeric

# Load csv data and test filesnya
def load_data(raw_path="data/raw"):
    train_df = pd.read_csv(os.path.join(raw_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(raw_path, "test.csv"))
    print(f"Loaded train shape: {train_df.shape}")
    print(f"Loaded test shape: {test_df.shape}")
    return train_df, test_df

# Remove duplicate rows from training set
def remove_duplicates(train_df):
    initial = train_df.shape[0]
    train_df = train_df.drop_duplicates()
    removed = initial - train_df.shape[0]
    print(f"Removed {removed} duplicate rows.")
    return train_df

# Save Student_ID from test dan hapus dari dataset
def extract_and_drop_ids(train_df, test_df, processed_path="data/processed"):
    os.makedirs(processed_path, exist_ok=True)

    id_col = None
    if "Student_ID" in test_df.columns:
        id_col = "Student_ID"

    if id_col:
        print(f"ID column found: '{id_col}'. Extracting and dropping.")
        test_ids = test_df[id_col].values
        test_df = test_df.drop(columns=[id_col])

        np.save(os.path.join(processed_path, "test_ids.npy"), test_ids)

        if id_col in train_df.columns:
            train_df = train_df.drop(columns=[id_col])
    else:
        print("WARNING: No ID column found.")

    return train_df, test_df

# Derived fitur-fiturnya
def add_derived_features(df):
    df["Grade_Trend"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    df["Approval_Trend"] = df["Curricular units 2nd sem (approved)"] - df["Curricular units 1st sem (approved)"]

    total_enrolled = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
    total_approved = df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
    df["Approval_Ratio"] = np.where(total_enrolled > 0, total_approved / total_enrolled, 0.0)

    df["Admission_Gap"] = df["Curricular units 1st sem (grade)"] - (df["Admission grade"] / 10.0)

    total_evaluations = df["Curricular units 1st sem (evaluations)"] + df["Curricular units 2nd sem (evaluations)"]
    df["Pressure_Indicator"] = np.where(total_enrolled > 0, total_evaluations / total_enrolled, 0.0)

    failed_sem1 = df["Curricular units 1st sem (enrolled)"] - df["Curricular units 1st sem (approved)"]
    failed_sem2 = df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 2nd sem (approved)"]
    df["Total_Failed_Credits"] = failed_sem1 + failed_sem2

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

    total_grades = df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]
    df["Avg_Grade"] = np.where(total_approved > 0, total_grades / 2, 0.0)

    df["Credit_Efficiency"] = np.where(total_evaluations > 0, total_approved / total_evaluations, 0.0)
    df["Academic_Engagement"] = total_enrolled + 0.5 * total_evaluations

    df["Zero_Sem1_Approved"] = (df["Curricular units 1st sem (approved)"] == 0).astype(int)
    df["Zero_Sem2_Approved"] = (df["Curricular units 2nd sem (approved)"] == 0).astype(int)

    df["Grade_Consistency"] = np.abs(
        df["Curricular units 1st sem (grade)"] - df["Curricular units 2nd sem (grade)"]
    )

    df["Econ_Stress_Index"] = df["Unemployment rate"] + df["Inflation rate"]
    df["Financial_Support_Gap"] = df["Scholarship holder"] - df["Debtor"]
    df["Financial_Risk"] = df["Debtor"] + (1 - df["Scholarship holder"]) + (1 - df["Tuition fees up to date"])
    df["GDP_Stress"] = -df["GDP"]

    # Demographic
    df["Age_Group"] = (df["Age at enrollment"] > 23).astype(int)

    df["Parent_Edu_Gap"] = np.abs(
        df["Mother's qualification"].astype(float) - df["Father's qualification"].astype(float)
    )
    df["Parent_Edu_Avg"] = (
        df["Mother's qualification"].astype(float) + df["Father's qualification"].astype(float)
    ) / 2

    low_edu_threshold = 10
    df["First_Gen_Risk"] = (
        (df["Mother's qualification"].astype(float) < low_edu_threshold) & 
        (df["Father's qualification"].astype(float) < low_edu_threshold)
    ).astype(int)

    df["Scholarship_Grade_Interaction"] = df["Scholarship holder"] * df["Avg_Grade"]
    df["Debtor_Failure_Interaction"] = df["Debtor"] * (df["Sem1_Failure_Rate"] + df["Sem2_Failure_Rate"])
    df["Age_Performance_Interaction"] = df["Age at enrollment"] * df["Approval_Ratio"]

    print(f"Derived features added: {23} new features including academic, socioeconomic, demographic, and interaction features.")
    return df

# Encode target  
def encode_data(train_df, test_df):
    target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
    if "Target" not in train_df.columns:
        raise ValueError("Target column not found")

    y_train = train_df["Target"].map(target_map).values
    X_train_raw = train_df.drop(columns=["Target"])
    X_test_raw = test_df.copy()
    # Kolom categorical
    nominal_cols = [
        "Marital status", "Application mode", "Course", "Previous qualification",
        "Nacionality", "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation"
    ]

    present_nominal = [c for c in nominal_cols if c in X_train_raw.columns]

    print(f"One-Hot Encoding Cols ({len(present_nominal)}): {present_nominal}")

    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    for col in present_nominal:
        combined[col] = combined[col].astype(str)

    combined_encoded = pd.get_dummies(combined, columns=present_nominal, dtype=int, drop_first=True)

    X_train_encoded = combined_encoded.iloc[: len(X_train_raw)]
    X_test_encoded = combined_encoded.iloc[len(X_train_raw) :]

    print(f"Features encoded. New feature count: {X_train_encoded.shape[1]}")

    return X_train_encoded, y_train, X_test_encoded

#  Shuffle split secara manual
def split_train_val(X, y, test_size=0.2, random_state=42):
    indices = np.arange(len(y))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    split_idx = int((1 - test_size) * len(y))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    return X.iloc[train_idx].values, X.iloc[val_idx].values, y[train_idx], y[val_idx]

# Untuk menangani missing value, fill NaNs dengan nilai rata2 dari training
def impute_missing_values(X_train, X_val, X_test):
    col_means = np.nanmean(X_train, axis=0)

    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_means, inds[1])

    inds = np.where(np.isnan(X_val))
    X_val[inds] = np.take(col_means, inds[1])

    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_means, inds[1])

    print("Missing values imputed using mean.")
    return X_train, X_val, X_test

# Clip based on train percentiles
def handle_outliers(X_train, X_val, X_test, method="clip", percentile=1):
    if method == "clip":
        lower = np.percentile(X_train, percentile, axis=0)
        upper = np.percentile(X_train, 100 - percentile, axis=0)

        X_train = np.clip(X_train, lower, upper)
        X_val = np.clip(X_val, lower, upper)
        X_test = np.clip(X_test, lower, upper)

        print(f"Outliers clipped at {percentile}th and {100-percentile}th percentile.")

    return X_train, X_val, X_test

# Standardized Z score
def standardize_features(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print("Features standardized using Z-score.")
    return X_train, X_val, X_test, mean, std


def select_features(X_train, X_val, X_test, correlation_threshold=0.95):
    # Remove constant columns
    print(f"\n=== Feature Selection (Threshold: {correlation_threshold}) ===")
    initial_count = X_train.shape[1]

    std = np.std(X_train, axis=0)
    constant_indices = np.where(std == 0)[0]
    
    if len(constant_indices) > 0:
        print(f"Dropping {len(constant_indices)} constant features.")
        X_train = np.delete(X_train, constant_indices, axis=1)
        X_val = np.delete(X_val, constant_indices, axis=1)
        X_test = np.delete(X_test, constant_indices, axis=1)

    # Hapus fitur dengan korelasi tinggi
    df_corr = pd.DataFrame(X_train)
    corr_matrix = df_corr.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features (> {correlation_threshold}).")
        X_train = np.delete(X_train, to_drop, axis=1)
        X_val = np.delete(X_val, to_drop, axis=1)
        X_test = np.delete(X_test, to_drop, axis=1)

    final_count = X_train.shape[1]
    print(f"Features reduced from {initial_count} to {final_count}.")
    
    return X_train, X_val, X_test

# Save all processed data
def save_processed_data(X_train, y_train, X_val, y_val, X_test, mean, std, processed_path="data/processed"):
    os.makedirs(processed_path, exist_ok=True)

    np.save(os.path.join(processed_path, "X_train.npy"), X_train)
    np.save(os.path.join(processed_path, "y_train.npy"), y_train)
    np.save(os.path.join(processed_path, "X_val.npy"), X_val)
    np.save(os.path.join(processed_path, "y_val.npy"), y_val)
    np.save(os.path.join(processed_path, "X_test_kaggle.npy"), X_test)
    np.save(os.path.join(processed_path, "scaler_mean.npy"), mean)
    np.save(os.path.join(processed_path, "scaler_std.npy"), std)

    print(f"All files saved to {processed_path}/")

# Run pipeline
def run_full_pipeline(raw_path="data/raw", processed_path="data/processed", handle_outliers_flag=True):
    print("=" * 60 + "\nSTARTING REFACTORED PIPELINE (One-Hot + CPU)\n" + "=" * 60)

    train_df, test_df = load_data(raw_path)
    train_df = remove_duplicates(train_df)
    train_df, test_df = extract_and_drop_ids(train_df, test_df, processed_path)

    train_df = add_derived_features(train_df)
    test_df = add_derived_features(test_df)

    X_train_df, y_train_all, X_test_df = encode_data(train_df, test_df)

    X_train, X_val, y_train, y_val = split_train_val(X_train_df, y_train_all)
    X_test = X_test_df.values

    X_train, X_val, X_test = impute_missing_values(X_train, X_val, X_test)

    if handle_outliers_flag:
        X_train, X_val, X_test = handle_outliers(X_train, X_val, X_test)

    X_train, X_val, X_test, mean, std = standardize_features(X_train, X_val, X_test)

    X_train, X_val, X_test = select_features(X_train, X_val, X_test, correlation_threshold=0.95)

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
    run_full_pipeline()