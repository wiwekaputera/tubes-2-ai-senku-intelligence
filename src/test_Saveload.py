# Test script buat coba save load
import numpy as np
import sys
import os

# Assuming dtl_scratch.py is in current directory or src/
try:
    from dtl_scratch import DecisionTreeScratch
except ImportError:
    sys.path.append('src')
    from dtl_scratch import DecisionTreeScratch

print("="*70)
print("TESTING DECISION TREE SAVE/LOAD FUNCTIONALITY")
print("="*70)

# Basic save/load test
print("\n" + "="*70)
print("TEST 1: Basic Save/Load with Simple Data")
print("="*70)

# Create simple synthetic data
np.random.seed(42)
X_simple = np.random.rand(100, 5)
y_simple = np.random.randint(0, 3, 100)

print(f"Training data shape: X={X_simple.shape}, y={y_simple.shape}")

# Train model
dt_original = DecisionTreeScratch(max_depth=5, min_samples_split=5, min_samples_leaf=2)
dt_original.fit(X_simple, y_simple)

# Get predictions before saving
y_pred_before = dt_original.predict(X_simple)
acc_before = np.mean(y_pred_before == y_simple)
print(f"\nAccuracy BEFORE save: {acc_before*100:.2f}%")

# Save model
save_path = "test_model.pkl"
dt_original.save(save_path)
print(f"\nModel saved to: {save_path}")
print(f"  File size: {os.path.getsize(save_path)} bytes")

# Load model
dt_loaded = DecisionTreeScratch.load(save_path)
print(f"Model loaded from: {save_path}")

# Get predictions after loading
y_pred_after = dt_loaded.predict(X_simple)
acc_after = np.mean(y_pred_after == y_simple)
print(f"\nAccuracy AFTER load: {acc_after*100:.2f}%")

# Verify predictions are identical
if np.array_equal(y_pred_before, y_pred_after):
    print("\n TEST 1 PASSED: Predictions are IDENTICAL!")
else:
    print("\n TEST 1 FAILED: Predictions differ!")
    diff_count = np.sum(y_pred_before != y_pred_after)
    print(f"   Different predictions: {diff_count}/{len(y_simple)}")

# Test save/load dengan dataset
print("\n" + "="*70)
print("TEST 2: Save/Load with Actual Data")
print("="*70)

try:
    # Try to load real data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_val = np.load("data/processed/y_val.npy")
    
    print(f" Real data loaded successfully")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    
    # Train model
    print("\nTraining model...")
    dt_real = DecisionTreeScratch(max_depth=6, min_samples_split=10, min_samples_leaf=5)
    dt_real.fit(X_train, y_train)
    
    y_train_pred_before = dt_real.predict(X_train)
    y_val_pred_before = dt_real.predict(X_val)
    
    train_acc_before = np.mean(y_train_pred_before == y_train)
    val_acc_before = np.mean(y_val_pred_before == y_val)
    
    print(f"\nBEFORE SAVE:")
    print(f"  Train Accuracy: {train_acc_before*100:.2f}%")
    print(f"  Val Accuracy:   {val_acc_before*100:.2f}%")
    
    # Save model
    save_path_real = "test_model_real.pkl"
    dt_real.save(save_path_real)
    print(f"\n Model saved to: {save_path_real}")
    print(f"  File size: {os.path.getsize(save_path_real):,} bytes")
    
    # Load model
    dt_real_loaded = DecisionTreeScratch.load(save_path_real)
    print(f" Model loaded from: {save_path_real}")
    
    y_train_pred_after = dt_real_loaded.predict(X_train)
    y_val_pred_after = dt_real_loaded.predict(X_val)
    
    train_acc_after = np.mean(y_train_pred_after == y_train)
    val_acc_after = np.mean(y_val_pred_after == y_val)
    
    print(f"\nAFTER LOAD:")
    print(f"  Train Accuracy: {train_acc_after*100:.2f}%")
    print(f"  Val Accuracy:   {val_acc_after*100:.2f}%")
    
    # Verify
    train_match = np.array_equal(y_train_pred_before, y_train_pred_after)
    val_match = np.array_equal(y_val_pred_before, y_val_pred_after)
    
    if train_match and val_match:
        print("\n TEST 2 PASSED: All predictions IDENTICAL!")
    else:
        print("\n TEST 2 FAILED: Predictions differ!")
        if not train_match:
            diff = np.sum(y_train_pred_before != y_train_pred_after)
            print(f"   Train differences: {diff}/{len(y_train)}")
        if not val_match:
            diff = np.sum(y_val_pred_before != y_val_pred_after)
            print(f"   Val differences: {diff}/{len(y_val)}")
    
except FileNotFoundError:
    print(" Data not found. Skipping TEST 2.")
    print(" Run preprocessing.py first to generate data.")

# Testing for tree
print("\n" + "="*70)
print("TEST 3: Tree Structure Integrity Check")
print("="*70)

# Use simple data from test 1
dt_check = DecisionTreeScratch(max_depth=3, min_samples_split=2, min_samples_leaf=1)
dt_check.fit(X_simple, y_simple)

# Count nodes before save
def count_nodes(node):
    if node['leaf']:
        return 1
    return 1 + count_nodes(node['left']) + count_nodes(node['right'])

nodes_before = count_nodes(dt_check.tree)
print(f"Tree nodes BEFORE save: {nodes_before}")

# Save and load
dt_check.save("test_structure.pkl")
dt_check_loaded = DecisionTreeScratch.load("test_structure.pkl")

nodes_after = count_nodes(dt_check_loaded.tree)
print(f"Tree nodes AFTER load:  {nodes_after}")

if nodes_before == nodes_after:
    print("\n TEST 3 PASSED: Tree structure preserved!")
else:
    print("\n TEST 3 FAILED: Tree structure corrupted!")

# Test Hyperparameters Preservation
print("\n" + "="*70)
print("TEST 4: Hyperparameters Preservation")
print("="*70)

# Create model 
dt_hyper = DecisionTreeScratch(max_depth=8, min_samples_split=15, min_samples_leaf=7)
dt_hyper.fit(X_simple, y_simple)

print("BEFORE SAVE:")
print(f"  max_depth:         {dt_hyper.max_depth}")
print(f"  min_samples_split: {dt_hyper.min_samples_split}")
print(f"  min_samples_leaf:  {dt_hyper.min_samples_leaf}")

# Save and load
dt_hyper.save("test_hyper.pkl")
dt_hyper_loaded = DecisionTreeScratch.load("test_hyper.pkl")

print("\nAFTER LOAD:")
print(f"  max_depth:         {dt_hyper_loaded.max_depth}")
print(f"  min_samples_split: {dt_hyper_loaded.min_samples_split}")
print(f"  min_samples_leaf:  {dt_hyper_loaded.min_samples_leaf}")

if (dt_hyper_loaded.max_depth == dt_hyper.max_depth and
    dt_hyper_loaded.min_samples_split == dt_hyper.min_samples_split and
    dt_hyper_loaded.min_samples_leaf == dt_hyper.min_samples_leaf):
    print("\n TEST 4 PASSED: Hyperparameters preserved!")
else:
    print("\n TEST 4 FAILED: Hyperparameters corrupted!")

# Test buat multiple save/load cycles
print("\n" + "="*70)
print("TEST 5: Multiple Save/Load Cycles")
print("="*70)

dt_cycle = DecisionTreeScratch(max_depth=4)
dt_cycle.fit(X_simple, y_simple)

y_pred_original = dt_cycle.predict(X_simple)

# Cycle 1
dt_cycle.save("test_cycle.pkl")
dt_cycle = DecisionTreeScratch.load("test_cycle.pkl")
y_pred_cycle1 = dt_cycle.predict(X_simple)

# Cycle 2
dt_cycle.save("test_cycle.pkl")
dt_cycle = DecisionTreeScratch.load("test_cycle.pkl")
y_pred_cycle2 = dt_cycle.predict(X_simple)

# Cycle 3
dt_cycle.save("test_cycle.pkl")
dt_cycle = DecisionTreeScratch.load("test_cycle.pkl")
y_pred_cycle3 = dt_cycle.predict(X_simple)

match1 = np.array_equal(y_pred_original, y_pred_cycle1)
match2 = np.array_equal(y_pred_original, y_pred_cycle2)
match3 = np.array_equal(y_pred_original, y_pred_cycle3)

print(f"Cycle 1 matches original: {match1}")
print(f"Cycle 2 matches original: {match2}")
print(f"Cycle 3 matches original: {match3}")

if match1 and match2 and match3:
    print("\n TEST 5 PASSED: Multiple save/load cycles successful!")
else:
    print("\n TEST 5 FAILED: Predictions degraded after multiple cycles!")

# Testing edge cases
print("\n" + "="*70)
print("TEST 6: Edge Cases")
print("="*70)

# Test 6a: Model with NaN in prediction
print("\nTest 6a: Predictions with NaN values")
X_nan = np.copy(X_simple[:10])
X_nan[0, 0] = np.nan  # Add NaN

dt_nan = DecisionTreeScratch(max_depth=3)
dt_nan.fit(X_simple, y_simple)
dt_nan.save("test_nan.pkl")
dt_nan_loaded = DecisionTreeScratch.load("test_nan.pkl")

try:
    y_pred_nan_before = dt_nan.predict(X_nan)
    y_pred_nan_after = dt_nan_loaded.predict(X_nan)
    
    if np.array_equal(y_pred_nan_before, y_pred_nan_after):
        print("NaN handling works after save/load")
    else:
        print("NaN handling failed after save/load")
except Exception as e:
    print(f"Error handling NaN: {e}")

# Test 6b: Very deep tree
print("\nTest 6b: Very deep tree")
dt_deep = DecisionTreeScratch(max_depth=20)
dt_deep.fit(X_simple, y_simple)

y_pred_deep_before = dt_deep.predict(X_simple)
dt_deep.save("test_deep.pkl")
dt_deep_loaded = DecisionTreeScratch.load("test_deep.pkl")
y_pred_deep_after = dt_deep_loaded.predict(X_simple)

if np.array_equal(y_pred_deep_before, y_pred_deep_after):
    print("Deep tree preserved correctly")
else:
    print("Deep tree corrupted")

# Cleaning up
print("\n" + "="*70)
print("CLEANUP")
print("="*70)

test_files = [
    "test_model.pkl",
    "test_model_real.pkl",
    "test_structure.pkl",
    "test_hyper.pkl",
    "test_cycle.pkl",
    "test_nan.pkl",
    "test_deep.pkl"
]

for file in test_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed: {file}")

# Summary from all test
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print("""
All tests completed!
""")
print("="*70)