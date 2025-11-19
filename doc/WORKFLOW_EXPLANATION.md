# Detailed Technical Workflow & Artifacts

This document explains exactly how data flows through our pipeline, what files are created at each step, and how the final submission is generated.

---

## Phase 1: Data Engineering Pipeline (Person 1)

**Script:** `src/preprocessing.py`  
**Goal:** Convert raw CSVs into clean, numerical NumPy arrays ready for model training.

### Inputs

- `data/raw/train.csv`: The training data (Features + Target + Student_ID).
- `data/raw/test.csv`: The Kaggle test data (Features + Student_ID). **NO TARGET**.

### Process Logic

1. **ID Handling (CRITICAL):**

   - **Train Data:** Drop the `Student_ID` column immediately. It is not a feature and will cause overfitting if included.
   - **Test Data:** Extract the `Student_ID` column and save it separately as `test_ids.npy` for the final submission. Then drop it from the features.

2. **Target Encoding:** Convert string labels to integers:

   - `"Dropout"` → `0`
   - `"Enrolled"` → `1`
   - `"Graduate"` → `2`

3. **Train/Validation Split:** Split `train.csv` into 80% Train and 20% Validation with random shuffling (seed=42).

4. **Imputation:** Handle missing values using mean imputation. **Fit on training data only**, then transform train, validation, and test sets.

5. **Feature Scaling (Standardization):** Apply Z-score normalization: $Z = \frac{X - \mu}{\sigma}$
   - Calculate mean ($\mu$) and standard deviation ($\sigma$) **from training data only**.
   - Apply the same transformation to validation and test sets.

### Outputs (Saved to `data/processed/`)

| File Name           | Shape (Approx) | Description                                    |
| ------------------- | -------------- | ---------------------------------------------- |
| `X_train.npy`       | (N_train, 36)  | Scaled features for training models.           |
| `y_train.npy`       | (N_train,)     | Integer labels (0, 1, 2) for training.         |
| `X_val.npy`         | (N_val, 36)    | Scaled features for validation.                |
| `y_val.npy`         | (N_val,)       | Integer labels (0, 1, 2) for validation.       |
| `X_test_kaggle.npy` | (N_test, 36)   | Scaled features from Kaggle test set (no IDs). |
| `test_ids.npy`      | (N_test,)      | Original Student IDs for submission mapping.   |
| `scaler_mean.npy`   | (36,)          | Mean values used for standardization.          |
| `scaler_std.npy`    | (36,)          | Std deviation values used for standardization. |

### Testing Your Work

Run `python src/preprocessing.py` and verify all `.npy` files are created in `data/processed/`.

---

## Phase 2: Model Implementation

### Person 2: Decision Tree (DTL)

**Script:** `src/dtl_scratch.py`  
**Goal:** Implement a Decision Tree classifier from scratch using NumPy.

#### Requirements

1. **Class Name:** `DecisionTreeScratch`
2. **Methods Required:**

   - `__init__(max_depth=None, min_samples_split=2, criterion='gini')`
   - `fit(X, y)` - Build the tree using training data
   - `predict(X)` - Return **NumPy array of integer predictions (0, 1, or 2)**

3. **Input/Output Specification:**

   - **fit() Input:**
     - `X`: NumPy array of shape `(n_samples, n_features)`
     - `y`: NumPy array of shape `(n_samples,)` with integer labels (0, 1, 2)
     - Returns: None (trains internally)
   - **predict() Input:**
     - `X`: NumPy array of shape `(n_samples, n_features)`
   - **predict() Output:**
     - NumPy array of shape `(n_samples,)` containing integers 0, 1, or 2

4. **Algorithm Details:**

   - Implement recursive tree building with stopping conditions
   - Support both Gini impurity and Entropy for splitting criteria
   - Must handle multi-class classification (3 classes)

5. **Testing:** Load the `.npy` files and verify your model can train and make predictions on validation data.

---

### Person 3: Logistic Regression

**Script:** `src/linear_models.py`  
**Goal:** Implement Logistic Regression from scratch and create a One-vs-All wrapper for multi-class classification.

#### Part A: Binary Logistic Regression

1. **Class Name:** `LogisticRegressionScratch`
2. **Methods Required:**
   - `__init__(learning_rate=0.01, n_iterations=1000)`
   - `fit(X, y)` - Train using gradient descent (y must be binary: 0 or 1)
   - `predict_proba(X)` - Return probability estimates
   - `predict(X)` - Return **NumPy array of binary predictions (0 or 1)**

#### Part B: One-vs-All Classifier (CRITICAL)

Since our target has **3 classes**, binary classifiers need a wrapper.

1. **Class Name:** `OneVsAllClassifier`
2. **Methods Required:**

   - `__init__(model_class, **kwargs)` - Takes a binary classifier class as input
   - `fit(X, y)` - Trains K binary classifiers (one per unique class)
   - `predict(X)` - Returns **NumPy array of integer predictions (0, 1, or 2)**

3. **Input/Output Specification:**

   - **fit() Input:**
     - `X`: NumPy array of shape `(n_samples, n_features)`
     - `y`: NumPy array of shape `(n_samples,)` with integer labels (0, 1, 2)
     - Returns: None
   - **predict() Input:**
     - `X`: NumPy array of shape `(n_samples, n_features)`
   - **predict() Output:**
     - NumPy array of shape `(n_samples,)` containing integers 0, 1, or 2

4. **How it Works:**

   - Train 3 separate binary classifiers (one for each class)
   - During prediction, pick the class with highest confidence

5. **Why This is Needed:**
   - Logistic Regression and SVM are binary classifiers
   - Our problem has 3 classes
   - **Person 4 (SVM) will also use this wrapper**

---

### Person 4: Support Vector Machine (SVM)

**Script:** `src/svm_scratch.py`  
**Goal:** Implement SVM from scratch using gradient descent or SMO algorithm.

#### Requirements

1. **Class Name:** `SVMScratch`
2. **Methods Required:**

   - `__init__(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)`
   - `fit(X, y)` - Train SVM (y must be binary: 0 or 1)
   - `predict(X)` - Return **NumPy array of binary predictions (0 or 1)**

3. **Input/Output Specification:**

   - **fit() Input:**
     - `X`: NumPy array of shape `(n_samples, n_features)`
     - `y`: NumPy array of shape `(n_samples,)` with binary labels (0, 1)
     - Returns: None
   - **predict() Input:**
     - `X`: NumPy array of shape `(n_samples, n_features)`
   - **predict() Output:**
     - NumPy array of shape `(n_samples,)` containing binary predictions (0, 1)

4. **Algorithm Details:**

   - Implement hinge loss: $L = \max(0, 1 - y \cdot f(x)) + \lambda ||w||^2$
   - Use gradient descent or SMO for optimization
   - **Your SVM is binary only**

5. **Multi-class Strategy:**
   - **Must use** the `OneVsAllClassifier` from `src/linear_models.py`
   - Wrap your SVM class with it to get 3-class predictions

---

## Phase 3: Integration & Submission (Person 5)

**Script:** `Tubes2_Notebook.ipynb`  
**Goal:** Load all models, compare performance, and generate the final Kaggle submission.

### Setup: Upload Data to Google Drive (For Colab Reproducibility)

**If running on Google Colab**, upload your raw CSV files to Google Drive first:

1. **Upload Files:**

   - Upload `train.csv` and `test.csv` to your Google Drive
   - Right-click each file → "Get link" → Change to "Anyone with the link can view"

2. **Extract File IDs:**

   - From the shareable link: `https://drive.google.com/file/d/1ZUtiaty9RPXhpz5F2Sy3dFPHF4YIt5iU/view?usp=sharing`
   - The File ID is: `1ZUtiaty9RPXhpz5F2Sy3dFPHF4YIt5iU`

3. **Notebook Import Cell:**

### Responsibilities

1. **Load Processed Data:**

   - Load all `.npy` files from `data/processed/`

2. **Train and Compare All Models:**

   - Import all three models from `src/`
   - Train each model on `X_train`, `y_train`
   - Get predictions on `X_val` (all models return integer arrays)
   - Calculate accuracy for each model

3. **Generate Kaggle Submission:**

   - Choose the best performing model
   - Predict on `X_test_kaggle.npy` (returns integer array)
   - Load `test_ids.npy`
   - **CRITICAL:** Convert integer predictions to strings:
     - `0` → `"Dropout"`
     - `1` → `"Enrolled"`
     - `2` → `"Graduate"`
   - Create DataFrame with columns: `Student_ID`, `Target`
   - Save as `submission.csv`

4. **Visualizations and Analysis:**
   - Create confusion matrices
   - Plot accuracy comparisons
   - Perform error analysis

### Final Submission Format

```csv
Student_ID,Target
2701,Dropout
1142,Graduate
3538,Enrolled
...
```
