"""
System B - SVM Training and Evaluation
=======================================
Loads the ResNet50 features extracted in system_b.py, performs an 80:20
train/test split, scales the features, tunes an SVM via 5-fold cross-validation,
and reports training/testing accuracy along with full evaluation metrics.
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── 1. Load Pre-extracted ResNet50 Features ───────────────────────
# Features were extracted once in system_b.py and saved to disk so we
# don't have to re-run the (expensive) ResNet50 forward pass every time.
print("Loading features from features_systemB.npz ...")
data = np.load("features_systemB.npz")
X, y = data["X"], data["y"]
print(f"Feature matrix shape : {X.shape}")
print(f"Labels shape         : {y.shape}")
print(f"Class balance        : happy(0)={np.sum(y == 0)}, angry(1)={np.sum(y == 1)}\n")

# ── 2. Train / Test Split (80:20) ─────────────────────────────────
# stratify=y keeps the happy/angry ratio balanced in both splits.
# random_state=42 ensures reproducibility for the report.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing  set: {X_test.shape[0]} samples\n")

# ── 3. Feature Scaling ────────────────────────────────────────────
# SVMs are sensitive to feature magnitudes, so we standardize features
# to zero mean / unit variance. IMPORTANT: fit the scaler ONLY on the
# training set, then apply the same transform to the test set. This
# prevents data leakage from test into training.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete (StandardScaler fit on train only).\n")

# ── 4. SVM Hyperparameter Tuning via GridSearchCV ─────────────────
# Linear-only fine-grained sweep. Previous experiments showed a ~80%
# performance ceiling regardless of kernel, with the best linear model
# (C=0.01) being the most stable (lowest std across CV folds).
# Here we do a fine logarithmic sweep around that region to confirm
# whether 80% is the true ceiling or if a sweet spot exists nearby.
param_grid = {
    "C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
}

print("Running GridSearchCV (5-fold CV) — linear-only fine sweep ...")
start = time.time()
grid = GridSearchCV(
    estimator=SVC(kernel="linear"),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,        # use all CPU cores
    verbose=1,
)
grid.fit(X_train_scaled, y_train)
elapsed = time.time() - start
print(f"\nGrid search finished in {elapsed:.1f} seconds.")
print(f"Best parameters     : {grid.best_params_}")
print(f"Best CV accuracy    : {grid.best_score_:.4f}\n")

best_svm = grid.best_estimator_

# Show ALL configurations from the grid search (only 11, so manageable).
results = pd.DataFrame(grid.cv_results_)
all_results = results.sort_values("mean_test_score", ascending=False)[
    ["params", "mean_test_score", "std_test_score"]
]
print("All hyperparameter combinations (sorted by CV accuracy):")
for i, row in all_results.iterrows():
    print(f"  CV={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})  {row['params']}")
print()

# ── 5. Evaluation Metrics ─────────────────────────────────────────
# Training accuracy: how well the model fits data it has seen.
# Testing accuracy : how well it generalizes to unseen data.
# The gap between them tells us about overfitting / underfitting.
y_train_pred = best_svm.predict(X_train_scaled)
y_test_pred = best_svm.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

print("=" * 55)
print("           SYSTEM B - FINAL RESULTS")
print("=" * 55)
print(f"Training Accuracy : {train_acc * 100:.2f}%")
print(f"Testing  Accuracy : {test_acc  * 100:.2f}%")
print(f"Train-Test Gap    : {gap * 100:.2f}%")

# Simple heuristic for fitting commentary in the report
if gap < 0.03:
    fit_quality = "Well-fitted (small gap between train and test accuracy)"
elif gap < 0.10:
    fit_quality = "Slightly overfitting (moderate gap)"
else:
    fit_quality = "Overfitting (large gap between train and test accuracy)"
print(f"Fitting Quality   : {fit_quality}")
print("=" * 55)

# ── 6. Detailed Classification Report & Confusion Matrix ──────────
print("\nClassification Report (Test Set):")
print(classification_report(
    y_test, y_test_pred,
    target_names=["happy", "angry"],
    digits=4,
))

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (Test Set):")
print("                 Predicted")
print("                happy  angry")
print(f"Actual happy    {cm[0, 0]:5d}  {cm[0, 1]:5d}")
print(f"Actual angry    {cm[1, 0]:5d}  {cm[1, 1]:5d}")


# ── 7. VISUALIZATIONS ───────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
C_values = [r["C"] for r in results["params"]]
mean_scores = results["mean_test_score"].tolist()
std_scores = results["std_test_score"].tolist()
 
 
# ── Chart 1: Grid Search CV Accuracy vs C Values ──────────────────
# Shows how each C value performed during cross validation
# Helps justify why C=0.005 was selected as the best parameter
fig1, ax1 = plt.subplots(figsize=(10, 5))
 
cv_df = pd.DataFrame({
    "C Value": [str(c) for c in C_values],
    "CV Accuracy": mean_scores,
    "Std": std_scores
}).sort_values("C Value")
 
sns.barplot(
    data=cv_df,x="C Value",y="CV Accuracy",palette="Blues_d",ax=ax1
)
 
# Add error bars for std deviation
ax1.errorbar(
    x=range(len(cv_df)),y=cv_df["CV Accuracy"],yerr=cv_df["Std"],fmt="none",color="black",capsize=4,linewidth=1.2
)
 
# Highlight the best bar
best_c_str = str(grid.best_params_["C"])
best_idx = list(cv_df["C Value"]).index(best_c_str)
ax1.patches[best_idx].set_facecolor("#e74c3c")  # Red for best bar
 
ax1.set_title("System B — Grid Search CV Accuracy vs C Values", fontsize=13, fontweight="bold")
ax1.set_xlabel("C Value (SVM Regularization)", fontsize=11)
ax1.set_ylabel("Mean CV Accuracy", fontsize=11)
ax1.set_ylim(0.70, 0.85)
ax1.axhline(y=grid.best_score_, color="red", linestyle="--", linewidth=1.2, label=f"Best CV = {grid.best_score_:.4f}")
ax1.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("gridsearch_cv_acc.png", dpi=150)
plt.show()
print("Chart 1 saved: gridsearch_cv_acc.png")
 
 
# ── Chart 2: Confusion Matrix Heatmap ─────────────────────────────
# Shows exactly how many happy/angry images were correctly classified
# and where the model made mistakes on the test set
fig2, ax2 = plt.subplots(figsize=(6, 5))
 
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",xticklabels=["Happy", "Angry"],yticklabels=["Happy", "Angry"],linewidths=0.5,linecolor="gray",annot_kws={"size": 14, "weight": "bold"},ax=ax2
)
 
ax2.set_title("System B — Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Predicted Label", fontsize=11)
ax2.set_ylabel("Actual Label", fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Chart 2 saved: confusion_matrix.png")
 
 
# ── Chart 3: Training vs Testing Accuracy Comparison ──────────────
# Directly answers the report question about fitting quality
# The gap between bars shows overfitting/underfitting/well-fitted
fig3, ax3 = plt.subplots(figsize=(6, 5))
 
acc_df = pd.DataFrame({
    "Split": ["Training Accuracy", "Testing Accuracy"],
    "Accuracy": [train_acc, test_acc]
})
 
bars = sns.barplot(
    data=acc_df, x="Split",y="Accuracy",palette=["#2ecc71", "#3498db"],ax=ax3
)
 
# Add value labels on top of each bar
for bar, val in zip(ax3.patches, [train_acc, test_acc]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val * 100:.2f}%",
        ha="center",va="bottom",fontsize=12, fontweight="bold"
    )
 
ax3.set_title("System B — Training vs Testing Accuracy", fontsize=13, fontweight="bold")
ax3.set_xlabel("")
ax3.set_ylabel("Accuracy", fontsize=11)
ax3.set_ylim(0, 1.05)
ax3.axhline(y=train_acc, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.5)
ax3.axhline(y=test_acc,  color="#3498db", linestyle="--", linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig("train_test_acc.png", dpi=150)
plt.show()
print("Chart 3 saved: train_test_acc.png")
