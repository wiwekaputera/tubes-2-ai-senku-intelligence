# IF3170 Machine Learning - Tugas Besar 2

**Topic:** Student Dropout Prediction (Academic Success Dataset)  
**Type:** From Scratch Implementation (NumPy Only) vs Scikit-Learn

---

## Project Paradigm

### 1. `src/` is the ENGINE

- Contains all the heavy logic, math, and class definitions (DecisionTree, SVM, Cleaning functions).
- Never write long execution scripts here.
- Think of these files as "Tools" in a toolbox.

### 2. `Tubes2_Notebook.ipynb` is the DRIVER

- This is where we import the tools from `src/` and use them.
- It orchestrates the flow: Load Data → Train Models → Compare Results → Generate Submission.
- All charts, analysis, and explanations go here.

---

## 🚀 Getting Started

We are using `uv` for dependency management.

### 1. Install `uv` (One time setup)

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**

```bash
curl -lsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone & Sync

Go to the project root folder and run:

```bash
uv sync
```

- This creates the `.venv` folder.
- It installs `numpy`, `pandas`, and `jupyter` with the exact same versions for everyone.

### 3. Run .py file
```bash
uv run path_to_file.py
```

### 4. How to Add New Libraries

If you need a new library (e.g., `matplotlib` or `seaborn`), **DO NOT** use `pip install`. Run:

```bash
uv add matplotlib seaborn
```

- Then commit the updated `uv.lock` and `pyproject.toml` file so everyone else gets it too.

---

## 🛠️ VSCode Setup for Notebooks

1. Open VSCode in this folder.
2. Open `Tubes2_Notebook.ipynb`.
3. **Select the Kernel** (Top Right):
   - Click "Select Kernel" / "Python 3...".
   - Select "Python Environments".
   - Choose the one marked `.venv`.

---

## 📂 Project Structure

```
Tubes_AI_IF3170/
│
├── data/
│   ├── raw/                    <-- train.csv and test.csv
│   └── processed/              <-- Generated .npy files go here (DO NOT EDIT)
│
├── src/                        <-- THE ENGINE (Logic & Classes)
│   ├── preprocessing.py        <-- Person 1: Data Pipeline
│   ├── dtl_scratch.py          <-- Person 2: Decision Tree Class
│   ├── linear_models.py        <-- Person 3: Logistic Regression & OvA
│   └── svm_scratch.py          <-- Person 4: SVM Class
│
├── Tugas_Besar_2_Group_XX.ipynb <-- THE DRIVER (Execution & Report)
├── main.py                     <-- Optional script for quick testing
├── pyproject.toml              <-- Dependency Config (Do not touch)
└── uv.lock                     <-- Version Lockfile (Do not touch)
```

---

## 🔄 Workflow Overview

### Phase 1: Data Engineering (Person 1)

- **Action:** Run `python src/preprocessing.py`.
- **Result:** Generates clean `.npy` files in `data/processed/`.
- **Note:** Everyone else needs these files to run their code.

### Phase 2: Model Development (Person 2, 3, 4)

- **Goal:** Build your class in `src/your_file.py`.
- **Testing:** Add a small `if __name__ == "__main__":` block at the bottom of your file to load the `.npy` data and test your model independently.

### Phase 3: Integration (Person 5)

- **Action:** Open the Notebook.
- **Logic:** Import classes from `src`, run the full training loop, visualize errors, and generate the Kaggle CSV.

---

## ⚠️ Rules

1. **DO NOT** push large CSVs to GitHub.
2. **ALWAYS** use `uv add <package>` instead of `pip install`.
3. **ALWAYS** run `uv sync` if you pull changes and the code crashes.
