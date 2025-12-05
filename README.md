# Student Dropout Prediction (Tubes 2 AI)

**Topic:** Academic Success Analysis & Dropout Prediction  
**Type:** From-Scratch Implementation (NumPy) vs Scikit-Learn Baseline

This repository contains the source code and documentation for the **IF3170 Artificial Intelligence** major assignment. The project focuses on predicting student dropout rates using machine learning algorithms implemented entirely from scratch, without relying on high-level libraries for the core logic.

---

## Team Members

**Group Number:** 02 (K03)

| NIM | Nama | Role & Responsibilities |
| :--- | :--- | :--- |
| 13523131 | Ahmad Wafi | SVM Implementation, Reporting |
| 13523143 | Amira Izani | Logistic Regression Implementation, Reporting |
| 13523147 | Frederiko Eldad Mugiyono | Integration, Hyperparameter Tuning, Submission |
| 13523157 | Natalia Desiany | Decision Tree Learning Implementation, Reporting |
| 13523160 | I Made Wiweka Putera | Data Cleaning, Preprocessing, Reporting |

---

## Project Architecture

The project is structured to separate core algorithmic logic from experimentation and reporting.

### 1. The Engine (`src/`)
Contains the core logic, mathematical computations, and class definitions.
*   **`preprocessing.py`**: Custom data pipeline including cleaning, feature engineering, and a from-scratch **SMOTE** implementation for class balancing.
*   **`dtl_scratch.py`**: Implementation of the **Decision Tree** classifier (CART algorithm) supporting Gini/Entropy criteria and pruning.
*   **`linear_models.py`**: Implementation of **Logistic Regression** using Mini-Batch Gradient Descent and **One-vs-All** wrapper for multi-class classification.
*   **`svm_scratch.py`**: Implementation of **Support Vector Machine (SVM)** using Gradient Descent optimization (Hinge Loss).

### 2. The Driver (`Tubes2_Notebook.ipynb`)
The main interface for the project. It orchestrates the entire workflow:
*   Loading and preprocessing data.
*   Training models with Grid Search Cross-Validation.
*   Visualizing performance (Confusion Matrices, Training Curves).
*   Comparing "Scratch" results vs Scikit-Learn.
*   Generating submission files.

---

## Installation & Usage

This project uses `uv` for fast and reliable dependency management.

### 1. Prerequisites
Ensure `uv` is installed on your system.

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -lsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment
Clone the repository and sync dependencies to create the virtual environment.

```bash
uv sync
```

### 3. Running the Code

**Option A: Jupyter Notebook (Recommended)**
Open `Tubes2_Notebook.ipynb` in VSCode or Jupyter Lab. Ensure the kernel is set to the `.venv` created by `uv`. This notebook contains the full analysis and final report.

**Option B: CLI Script**
For quick training and testing without the notebook interface:

```bash
uv run main.py
```

To run only the data preprocessing pipeline:
```bash
uv run src/preprocessing.py
```

---

## 📂 Repository Structure

```
Tubes_AI_IF3170/
│
├── data/
│   ├── raw/                    <-- Original datasets (train.csv, test.csv)
│   ├── processed/              <-- Transformed .npy files (SMOTE balanced)
│   └── submit/                 <-- Generated submission CSVs
│
├── doc/                        <-- Comparisons and analytical reports
│
├── src/                        <-- Source code for algorithms
│   ├── preprocessing.py
│   ├── dtl_scratch.py
│   ├── linear_models.py
│   └── svm_scratch.py
│
├── Tubes2_Notebook.ipynb       <-- Primary project notebook
├── main.py                     <-- CLI entry point
├── pyproject.toml              <-- Dependency configuration
└── uv.lock                     <-- Lockfile for reproducible builds
```

---

## Methodology

1.  **Data Engineering**:
    *   Handling missing values and outliers.
    *   Feature engineering (academic, socioeconomic, and demographic indicators).
    *   Addressing class imbalance using a custom **SMOTE** (Synthetic Minority Over-sampling Technique) implementation.

2.  **Model Development**:
    *   All models are implemented as Python classes following the Scikit-Learn `fit`/`predict` interface API style for consistency.
    *   Optimization techniques include **Mini-Batch Gradient Descent**, **Learning Rate Decay**, and **Early Stopping**.

3.  **Evaluation**:
    *   Models are validated using **5-Fold Cross-Validation**.
    *   Hyperparameters are tuned via **Grid Search**.
    *   Final performance is benchmarked against Scikit-Learn's equivalent estimators to verify correctness.
