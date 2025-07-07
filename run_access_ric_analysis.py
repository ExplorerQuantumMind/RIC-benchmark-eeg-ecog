#!/usr/bin/env python3
# simulate_lfp_rf.py

import pyodbc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# 1) Configuration
DB_PATH = r"C:\Users\mahsa\OneDrive\Desktop\newa\ric_data.accdb"
LFP_TABLES = [f"LFP_ch{i}" for i in range(1,6)]
FEATURE_STEPS = [
    ['curvature'],
    ['curvature','complexity'],
    ['curvature','complexity','recursion_level'],
    ['curvature','complexity','recursion_level','geometry_variability'],
    ['curvature','complexity','recursion_level','geometry_variability','temporal_dynamics'],
    ['curvature','complexity','recursion_level','geometry_variability','temporal_dynamics','gravitational_modulation'],
    ['curvature','complexity','recursion_level','geometry_variability','temporal_dynamics','gravitational_modulation','topological_changes'],
    ['curvature','complexity','recursion_level','geometry_variability','temporal_dynamics','gravitational_modulation','topological_changes','environmental_responsivity'],
    ['curvature','complexity','recursion_level','geometry_variability','temporal_dynamics','gravitational_modulation','topological_changes','environmental_responsivity','collapse_entropy','interaction_features']
]

# 2) Connect and load LFP channels
conn = pyodbc.connect(
    r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
    rf"DBQ={DB_PATH};"
)
data = {}
for tbl in LFP_TABLES:
    df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
    # grab first numeric column
    num = df.select_dtypes(include='number').columns[0]
    data[tbl] = df[num].values.astype(float)
conn.close()

# 3) Build a matrix (channels × samples)
lfp_matrix = np.vstack([data[t] for t in LFP_TABLES])
n_ch, _ = lfp_matrix.shape

# 4) Compute RIC‐style features per channel
def compute_features(mat):
    feats = {f:[] for f in FEATURE_STEPS[-1]}
    for sig in mat:
        # simple stand‐ins for your 10 metrics:
        feats['curvature'].append(np.var(sig))                 # curvature ~ variance
        feats['complexity'].append(np.mean(np.abs(np.diff(sig))))  # complexity ~ mean abs diff
        feats['recursion_level'].append(np.std(sig))           # recursion_level ~ std
        feats['geometry_variability'].append(np.mean(sig**2))  # geo_var ~ mean power
        feats['temporal_dynamics'].append(np.mean(np.diff(sig)**2))
        feats['gravitational_modulation'].append(np.max(sig)-np.min(sig))
        feats['topological_changes'].append(np.sum(sig>np.mean(sig)))
        feats['environmental_responsivity'].append(np.corrcoef(sig, np.arange(sig.size))[0,1])
        # collapse_entropy: approximate Shannon entropy of normalized PSD
        psd = np.abs(np.fft.rfft(sig))**2
        psd /= psd.sum()+1e-12
        feats['collapse_entropy'].append(-np.sum(psd*np.log2(psd+1e-12)))
        # interaction_features: simple product of two
        feats['interaction_features'].append(feats['curvature'][-1]*feats['collapse_entropy'][-1])
    return pd.DataFrame(feats)

df_feats = compute_features(lfp_matrix)

# 5) Simulate binary label: high collapse_entropy ⇒ 0, else 1
ce = df_feats['collapse_entropy'].values
prob1 = 1/(1+np.exp(ce- np.median(ce)))  # sigmoid around median
labels = np.random.binomial(1, prob1)
df_feats['label'] = labels

# 6) Random Forest over incremental feature sets
results = []
y = df_feats['label'].values

for feats in FEATURE_STEPS:
    X = df_feats[feats].values
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    clf.fit(X, y)
    yprob = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, yprob)
    fpr, tpr, t = roc_curve(y, yprob)
    j = tpr - fpr
    o = t[np.argmax(j)]
    pred = (yprob>=o).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    results.append({
        'Feature Set': ' + '.join(feats),
        'AUC': round(auc,2),
        'Threshold': round(o,2),
        'Sensitivity': round(tp/(tp+fn),2),
        'Specificity': round(tn/(tn+fp),2)
    })

# 7) Print results
df_res = pd.DataFrame(results)
print("\nRandom Forest Performance Across LFP Channels (simulated labels):\n")
print(df_res.to_string(index=False))
