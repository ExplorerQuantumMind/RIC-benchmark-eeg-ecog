#!/usr/bin/env python3
# simulate_lfp_rf.py

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try importing pyodbc, with user-friendly error
try:
    import pyodbc
except ModuleNotFoundError:
    sys.exit("❌ pyodbc not found. Please `pip install pyodbc` and ensure the Access ODBC driver is installed.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ——— 1) Load channels from Access ———
DB_PATH = r"C:\Users\mahsa\OneDrive\Desktop\newa\ric_data.accdb"
conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    rf"DBQ={DB_PATH};"
)
print(f"Connecting to Access DB at:\n  {DB_PATH}\n")

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()
lfp_tables = sorted(
    row.table_name for row in cursor.tables(tableType='TABLE')
    if row.table_name.lower().startswith("lfp_ch")
)
conn.close()

if not lfp_tables:
    sys.exit("❌ No tables named LFP_ch* found in the database.")

print(f"✅ Found {len(lfp_tables)} LFP tables: {lfp_tables}\n")

# Load numeric series
data = []
conn = pyodbc.connect(conn_str)
for tbl in lfp_tables:
    df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) == 0:
        sys.exit(f"❌ No numeric columns in table {tbl}")
    sig = df[num_cols[0]].astype(float).to_numpy()
    data.append(sig)
    print(f"  • Loaded {tbl}: {sig.size} samples")
conn.close()
print()

# Stack channels
X_raw = np.vstack(data)
n_ch, n_t = X_raw.shape
print(f"✅ Assembled matrix: {n_ch} channels × {n_t} samples\n")

# ——— 2) Compute RIC‐style features ———
def compute_features(mat):
    feats = []
    for sig in mat:
        curvature = np.var(sig)
        complexity = np.mean(np.abs(np.diff(sig)))
        recursion_level = np.std(sig)
        geometry_variability = np.mean(sig**2)
        temporal_dynamics = np.mean(np.diff(sig)**2)
        gravitational_modulation = np.max(sig)-np.min(sig)
        topological_changes = np.sum(sig>sig.mean())
        env_resp = np.corrcoef(sig, np.arange(len(sig)))[0,1]
        psd = np.abs(np.fft.rfft(sig))**2
        psd_norm = psd/psd.sum()
        collapse_entropy = -np.sum(psd_norm*np.log2(psd_norm+1e-12))
        interaction = curvature*collapse_entropy
        feats.append({
            'curvature': curvature,
            'complexity': complexity,
            'recursion_level': recursion_level,
            'geometry_variability': geometry_variability,
            'temporal_dynamics': temporal_dynamics,
            'gravitational_modulation': gravitational_modulation,
            'topological_changes': topological_changes,
            'environmental_responsivity': env_resp,
            'collapse_entropy': collapse_entropy,
            'interaction_features': interaction
        })
    return pd.DataFrame(feats)

df = compute_features(X_raw)
print("✅ Computed features:\n", df.head(), "\n")

# ——— 3) Simulate labels ———
ce = df['collapse_entropy'].values
prob = 1/(1 + np.exp(ce - np.median(ce)))
labels = np.random.binomial(1, prob)
df['label'] = labels
print("✅ Labels simulated.\n")

# ——— 4) Hyperparameter tuning ———
features = df.columns[:-1]
X = df[features].values
y = df['label'].values

loo = LeaveOneOut()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, scoring='roc_auc', cv=loo, n_jobs=-1)
grid.fit(X, y)
best = grid.best_params_
print("✅ Best params:", best, "\n")

# ——— 5) Cross‐validated ROC ———
clf = RandomForestClassifier(**best, random_state=42)
y_proba = cross_val_predict(clf, X, y, cv=loo, method='predict_proba')[:,1]
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LOOCV ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ——— 6) Feature importances ———
clf_full = RandomForestClassifier(**best, random_state=42)
clf_full.fit(X, y)
imp = clf_full.feature_importances_

plt.figure()
plt.bar(np.arange(len(imp)), imp)
plt.xticks(np.arange(len(imp)), features, rotation=45, ha='right')
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

print("\n✅ All analyses complete.") 
