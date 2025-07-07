#!/usr/bin/env python3
# ecog_mat_ric_rf.py

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc

# 1) Find all ECoG_ch*.mat in cwd
mat_files = sorted(glob.glob(os.path.join(os.getcwd(), "ECoG_ch*.mat")))
if not mat_files:
    sys.exit("❌ No files matching ECoG_ch*.mat found in this folder.")
print(f"🔍 Found {len(mat_files)} .mat files:")
for f in mat_files:
    print("  ", os.path.basename(f))
print()

# 2) Load each file into a 1D signal (skip empties)
signals = []
for m in mat_files:
    data = loadmat(m)
    sig = None
    for v in data.values():
        arr = np.asarray(v)
        if arr.ndim == 1 and arr.size > 0:
            sig = arr.astype(float)
            break
        if arr.ndim == 2 and 1 in arr.shape and arr.size > 1:
            sig = arr.flatten().astype(float)
            break
    if sig is None:
        print(f"⚠️  Skipping {os.path.basename(m)} (no nonempty 1D data)")
        continue
    signals.append(sig)
    print(f"  • Loaded {os.path.basename(m)} ({sig.size} samples)")
if not signals:
    sys.exit("❌ No valid signals after skipping empties.")
print(f"\n✅ Ready to process {len(signals)} channels.\n")

# 3) Compute RIC features per signal
def ric_features(sig):
    curvature                = np.var(sig)
    complexity               = np.mean(np.abs(np.diff(sig))) if sig.size>1 else 0.0
    recursion_level          = np.std(sig)
    geometry_variability     = np.mean(sig**2)
    temporal_dynamics        = np.mean(np.diff(sig)**2) if sig.size>1 else 0.0
    gravitational_modulation = np.ptp(sig)
    topological_changes      = np.sum(sig > sig.mean())
    env_resp                 = np.corrcoef(sig, np.arange(len(sig)))[0,1] \
                               if sig.size>1 else 0.0
    psd                      = np.abs(np.fft.rfft(sig))**2
    psd_norm                 = psd / psd.sum()
    collapse_entropy         = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    interaction_features     = curvature * collapse_entropy
    return {
        'curvature': curvature,
        'complexity': complexity,
        'recursion_level': recursion_level,
        'geometry_variability': geometry_variability,
        'temporal_dynamics': temporal_dynamics,
        'gravitational_modulation': gravitational_modulation,
        'topological_changes': topological_changes,
        'environmental_responsivity': env_resp,
        'collapse_entropy': collapse_entropy,
        'interaction_features': interaction_features
    }

rows = [ric_features(sig) for sig in signals]
df = pd.DataFrame(rows)
print("Feature matrix (first 5 rows):\n", df.head(), "\n")

# 4) Simulate binary labels from collapse_entropy
ce = df['collapse_entropy'].values
prob = 1 / (1 + np.exp(ce - np.median(ce)))
df['label'] = np.random.binomial(1, prob)
print("Label counts:\n", df['label'].value_counts(), "\n")

# 5) Hyperparameter tuning via LOOCV
X = df.drop(columns='label').values
y = df['label'].values
loo = LeaveOneOut()
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [2,3,5],
    'min_samples_leaf': [1,2,4]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, scoring='roc_auc', cv=loo, n_jobs=-1)
grid.fit(X, y)
best = grid.best_params_
print("🔧 Best RF params:", best, "\n")

# 6) Plot LOOCV ROC
clf = RandomForestClassifier(**best, random_state=42)
y_proba = cross_val_predict(clf, X, y, cv=loo, method='predict_proba')[:,1]
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LOOCV ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 7) Plot feature importances
clf_full = RandomForestClassifier(**best, random_state=42)
clf_full.fit(X, y)
imp = clf_full.feature_importances_

plt.figure(figsize=(8,4))
plt.bar(df.columns[:-1], imp)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

print("✅ Analysis complete.")
