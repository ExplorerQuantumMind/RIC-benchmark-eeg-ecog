#!/usr/bin/env python3
import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 1) Auto-discover the .accdb file in cwd
dbs = glob.glob(os.path.join(os.getcwd(), "*.accdb"))
if not dbs:
    sys.exit("❌ No .accdb file found in this folder. Please place your Access DB here.")
DB_PATH = dbs[0]
print(f"Using database: {DB_PATH}\n")

# 2) Load ECoG_ch* tables
try:
    import pyodbc
except ModuleNotFoundError:
    sys.exit("❌ pyodbc not installed. Run `pip install pyodbc`.")

conn_str = (
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    rf"DBQ={DB_PATH};"
)
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()
tables = sorted(
    row.table_name for row in cursor.tables(tableType='TABLE')
    if row.table_name.lower().startswith("ecog_ch")
)
conn.close()

if not tables:
    sys.exit("❌ No tables named ECoG_ch* found in the DB.")

print(f"Found {len(tables)} channels:", tables, "\n")

# 3) Read each channel's numeric column
channels = []
conn = pyodbc.connect(conn_str)
for tbl in tables:
    df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
    num_cols = df.select_dtypes(include='number').columns
    if not len(num_cols):
        sys.exit(f"❌ Table {tbl} has no numeric columns.")
    sig = df[num_cols[0]].astype(float).to_numpy()
    channels.append(sig)
    print(f"  • Loaded {tbl} ({sig.size} samples)")
conn.close()
print()

# 4) Stack into matrix and compute features
mat = np.vstack(channels)
print(f"{len(channels)} channels × {mat.shape[1]} timepoints\n")

def compute_features(mat):
    feats = []
    for sig in mat:
        curvature                = np.var(sig)
        complexity               = np.mean(np.abs(np.diff(sig)))
        recursion_level          = np.std(sig)
        geometry_variability     = np.mean(sig**2)
        temporal_dynamics        = np.mean(np.diff(sig)**2)
        gravitational_modulation = np.ptp(sig)
        topological_changes      = np.sum(sig>sig.mean())
        env_resp                 = np.corrcoef(sig, np.arange(sig.size))[0,1]
        psd                      = np.abs(np.fft.rfft(sig))**2
        psd_norm                 = psd/psd.sum()
        collapse_entropy         = -np.sum(psd_norm*np.log2(psd_norm+1e-12))
        interaction_features     = curvature * collapse_entropy
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
            'interaction_features': interaction_features
        })
    return pd.DataFrame(feats)

df = compute_features(mat)
print("Feature matrix (first 5 rows):\n", df.head(), "\n")

# 5) Simulate binary labels from collapse_entropy
ce = df['collapse_entropy'].values
prob = 1/(1 + np.exp(ce - np.median(ce)))
df['label'] = np.random.binomial(1, prob)
print("Labels simulated:\n", df['label'].value_counts(), "\n")

# 6) Hyperparameter tuning via LOOCV
X = df.drop(columns='label').values
y = df['label'].values

loo = LeaveOneOut()
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [2,3,5],
    'min_samples_leaf': [1,2,4]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                    scoring='roc_auc', cv=loo, n_jobs=-1)
grid.fit(X, y)
best = grid.best_params_
print("Best RF params:", best, "\n")

# 7) Plot LOOCV ROC
clf = RandomForestClassifier(**best, random_state=42)
y_proba = cross_val_predict(clf, X, y, cv=loo, method='predict_proba')[:,1]
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LOOCV ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 8) Plot feature importances
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
