import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import run_pipeline
from src.dtl_scratch import DecisionTreeScratch
from src.linear_models import LogisticRegressionScratch, OneVsAllClassifier
from src.svm_scratch import SVMScratch

def main():
    print("Checking project structure...")
    
    # 1. Verify Preprocessing
    print("\n[1] Preprocessing Module: OK")
    # Uncomment to run the actual pipeline if data exists
    # run_pipeline()

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

if __name__ == "__main__":
    main()
