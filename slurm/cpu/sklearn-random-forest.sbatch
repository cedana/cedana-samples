#!/bin/bash
#SBATCH --job-name=sklearn-rf
#SBATCH --output=sklearn-rf-%j.out
#SBATCH --error=sklearn-rf-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Workload: Scikit-learn Random Forest Training
# Description: Trains Random Forest models on synthetic datasets with varying sizes
# Type: CPU

echo "Starting Scikit-learn Random Forest workload on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." || exit 1

echo "Installing dependencies..."
pip install scikit-learn pandas numpy joblib --quiet

python3 << 'EOF'
"""
Scikit-learn Random Forest Training Benchmark

Trains Random Forest models on synthetic datasets of varying sizes.
Random Forest builds many decision trees in parallel, maintaining:
- Individual tree structures
- Split decisions and thresholds
- Feature importance statistics

The embarrassingly parallel nature and large model state make this
ideal for checkpoint/restore testing.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import time
import sys
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Scikit-learn Random Forest Benchmark")
print("=" * 70)
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")
print()

def train_random_forest_classifier(n_samples, n_features, n_estimators, max_depth):
    """Train a Random Forest classifier."""
    print(f"\n--- Random Forest Classifier ---")
    print(f"    Samples: {n_samples:,} | Features: {n_features} | "
          f"Trees: {n_estimators} | Max Depth: {max_depth}")

    # Generate data
    print("  Generating synthetic data...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_clusters_per_class=3,
        n_classes=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Train model
    print(f"  Training {n_estimators} trees...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=4,
        random_state=42,
        verbose=1,
        warm_start=False
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"  Training time: {train_time:.2f}s")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test F1 Score: {f1:.4f}")

    # Feature importance
    top_features = np.argsort(model.feature_importances_)[-5:][::-1]
    print(f"  Top 5 important features: {top_features}")

    return model, train_time

def train_random_forest_regressor(n_samples, n_features, n_estimators, max_depth):
    """Train a Random Forest regressor."""
    print(f"\n--- Random Forest Regressor ---")
    print(f"    Samples: {n_samples:,} | Features: {n_features} | "
          f"Trees: {n_estimators} | Max Depth: {max_depth}")

    # Generate data
    print("  Generating synthetic data...")
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        noise=0.1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    print(f"  Training {n_estimators} trees...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=4,
        random_state=42,
        verbose=1
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  Training time: {train_time:.2f}s")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test R²: {r2:.4f}")

    return model, train_time

def train_extra_trees(n_samples, n_features, n_estimators):
    """Train Extra Trees (Extremely Randomized Trees)."""
    print(f"\n--- Extra Trees Classifier ---")
    print(f"    Samples: {n_samples:,} | Features: {n_features} | Trees: {n_estimators}")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_classes=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        n_jobs=4,
        random_state=42,
        verbose=1
    )

    print(f"  Training {n_estimators} extremely randomized trees...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    accuracy = model.score(X_test, y_test)
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Test Accuracy: {accuracy:.4f}")

    return model, train_time

def train_gradient_boosting(n_samples, n_features, n_estimators):
    """Train Gradient Boosting (sequential tree building)."""
    print(f"\n--- Gradient Boosting Classifier ---")
    print(f"    Samples: {n_samples:,} | Features: {n_features} | Trees: {n_estimators}")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Note: GradientBoosting is sequential, not parallel
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbose=1
    )

    print(f"  Training {n_estimators} boosted trees (sequential)...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    accuracy = model.score(X_test, y_test)
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Test Accuracy: {accuracy:.4f}")

    return model, train_time

# ============================================================
# Run Benchmarks
# ============================================================
results = []

# Small dataset, many trees
_, t = train_random_forest_classifier(50_000, 50, 500, 20)
results.append(('rf_clf_50k_500trees', t))

# Medium dataset
_, t = train_random_forest_classifier(200_000, 100, 300, 25)
results.append(('rf_clf_200k_300trees', t))

# Large dataset, deeper trees
_, t = train_random_forest_classifier(500_000, 100, 200, 30)
results.append(('rf_clf_500k_200trees', t))

# Regression
_, t = train_random_forest_regressor(200_000, 50, 300, 20)
results.append(('rf_reg_200k_300trees', t))

# Extra Trees (faster due to random splits)
_, t = train_extra_trees(300_000, 100, 500)
results.append(('extra_trees_300k', t))

# Gradient Boosting (sequential, stateful)
_, t = train_gradient_boosting(100_000, 50, 200)
results.append(('grad_boost_100k', t))

# ============================================================
# Summary
# ============================================================
print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
total_time = 0
for name, elapsed in results:
    print(f"  {name:30s}: {elapsed:8.2f}s")
    total_time += elapsed
print("-" * 70)
print(f"  {'TOTAL':30s}: {total_time:8.2f}s")
print("=" * 70)
print("\nRandom Forest benchmark complete!")
EOF

echo "Job completed at: $(date)"
