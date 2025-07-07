#!/usr/bin/env python3
# ecog_rf_fast.py
# 3‐Fold RF on 3 RIC features + optimal threshold & metrics

import glob, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, accuracy_score,
    precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier

# 1) Locate and load .mat channels
mat_files = sorted(glob.glob("ECoG_ch*.mat"))
if not mat_files:
    sys.exit("❌ No ECoG_ch*.mat files found in current folder.")

signals = []
for fname in mat_files:
    data = loadmat(fname)
    # grab first 1D or single‐column array
    arr = next(v for v in data.values()
               if isinstance(v, np.ndarray) and v.size > 1)
    sig = arr.flatten().astype(float)
    signals.append(sig)
print(f"✅ Loaded {len(signals)} channels")

# 2) Extract three RIC features: variance, peak‐to‐peak, spectral entropy
def spectral_entropy(x):
    P = np.abs(np.fft.rfft(x))**2
    P_norm = P / (P.sum() + 1e-12)
    return -np.sum(P_norm * np.log2(P_norm + 1e-12))

X = []
for sig in signals:
    var = np.var(sig)
    ptp = np.ptp(sig)
    ent = spectral_entropy(sig)
    X.append([var, ptp, ent])
X = np.array(X)

# 3) Median‐split labels on entropy
y = (X[:,2] > np.median(X[:,2])).astype(int)
print("Labels 0/1 counts:", np.bincount(y))

# 4) Build a fast RF pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("skb", SelectKBest(f_classif, k=2)),
    ("clf", RandomForestClassifier(random_state=0))
])

param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [3, 5]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
grid = GridSearchCV(
    pipe, param_grid,
    cv=cv, scoring="roc_auc",
    verbose=2, n_jobs=1
)

# 5) Fit
grid.fit(X, y)
print(f"\n🏆 Best RF AUC = {grid.best_score_:.2f}")
print("Best params:", grid.best_params_)

# 6) Generate cross‐validated probabilities
y_proba = cross_val_predict(
    grid.best_estimator_, X, y,
    cv=cv, method="predict_proba"
)[:, 1]

# 7) Compute ROC & plot
fpr, tpr, thresholds = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("3-Fold RF ROC (Fast)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# === NEW: optimal threshold & classification metrics ===
J = tpr - fpr
ix = np.argmax(J)
opt_thr = thresholds[ix]
print(f"\n✔ Optimal threshold = {opt_thr:.3f} (Youden’s J = {J[ix]:.3f})")

y_pred = (y_proba >= opt_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy    = accuracy_score(y, y_pred)
precision   = precision_score(y, y_pred, zero_division=0)
recall      = recall_score(y, y_pred, zero_division=0)
f1          = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("🔍 Prediction performance at optimal threshold:")
print(f"  • Sensitivity (Recall): {sensitivity:.3f}")
print(f"  • Specificity        : {specificity:.3f}")
print(f"  • Accuracy           : {accuracy:.3f}")
print(f"  • Precision          : {precision:.3f}")
print(f"  • F1 Score           : {f1:.3f}")

print("\n✅ Done.")
