#!/usr/bin/env python3
# ecog_multi_db_ric_rf.py

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 1) Find all channel‐db files
db_files = sorted(glob.glob(os.path.join(os.getcwd(), "ECoG_ch*.accdb")))
if not db_files:
    sys.exit("❌ No files matching ECoG_ch*.accdb found in this folder.")

print(f"🔍 Found {len(db_files)} channel files:")
for f in db_files:
    print("  ", os.path.basename(f))
print()

# 2) Load each channel’s numeric series
signals = []
for db in db_files:
    conn_str = (
        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
        rf"DBQ={db};"
    )
    try:
        import pyodbc
    except ModuleNotFoundError:
        sys.exit("❌ pyodbc missing. `pip install pyodbc` and retry.")
    conn = pyodbc.connect(conn_str)
    # find the first table in that file
    tbls = [r.table_name for r in conn.cursor().tables(tableType='TABLE')]
    if not tbls:
        conn.close()
        sys.exit(f"❌ No tables found inside {os.path.basename(db)}")
    tbl = tbls[0]
    df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
    conn.close()

    # pick the first numeric column
    numcols = df.select_dtypes('number').columns
    if len(numcols)==0:
        sys.exit(f"❌ Table {tbl} in {os.path.basename(db)} has no numeric column.")
    sig = df[numcols[0]].astype(float).to_numpy()
    signals.append(sig)
    print(f"  • {os.path.basename(db)} → Table '{tbl}' → {sig.size} samples")
print()

# stack
X_raw = np.vstack(signals)
print(f"✅ Assembled data: {len(signals)} channels × {X_raw.shape[1]} samples\n")

# 3) Compute RIC‐style features
def compute_features(mat):
    rows = []
    for sig in mat:
        curvature                = np.var(sig)
        complexity               = np.mean(np.abs(np.diff(sig)))
        recursion_level          = np.std(sig)
        geometry_variability     = np.mean(sig**2)
        temporal_dynamics        = np.mean(np.diff(sig)**2)
        gravitational_modulation = np.ptp(sig)
        topological_changes      = np.sum(sig>sig.mean())
        env_resp                 = np.corrcoef(sig, np.arange(len(sig)))[0,1]
        psd                      = np.abs(np.fft.rfft(sig))**2
        psd_norm                 = psd/psd.sum()
        collapse_entropy         = -np.sum(psd_norm * np.log2(psd_norm+1e-12))
        interaction_features     = curvature * collapse_entropy
        rows.append({
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
        })
    return pd.DataFrame(rows)

df = compute_features(X_raw)
print("Feature matrix (first 5 rows):\n", df.head(), "\n")

# 4) Simulate binary labels
ce = df['collapse_entropy'].values
prob = 1/(1 + np.exp(ce - np.median(ce)))
df['label'] = np.random.binomial(1, prob)
print("Simulated labels count:\n", df['label'].value_counts(), "\n")

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

print("✅ All done.")
