#!/usr/bin/env python3
# ecog_advanced_fixed.py

import os, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 1) find .mat files
mat_files = sorted(glob.glob(os.path.join(os.getcwd(), "ECoG_ch*.mat")))
if not mat_files:
    sys.exit("❌ No ECoG_ch*.mat files found.")
print(f"🔍 Found {len(mat_files)} channels\n")

# 2) load signals
signals = []
for m in mat_files:
    data = loadmat(m)
    sig = None
    for v in data.values():
        arr = np.asarray(v)
        if arr.ndim == 1 and arr.size > 1:
            sig = arr.astype(float); break
        if arr.ndim == 2 and 1 in arr.shape and arr.size > 1:
            sig = arr.flatten().astype(float); break
    if sig is None:
        print(f"⚠️  Skipping empty {os.path.basename(m)}")
        continue
    signals.append(sig)
    print(f"  • {os.path.basename(m)}: {sig.size} samples")
print(f"\n✅ Loaded {len(signals)} non-empty channels\n")

# 3) feature functions
def hjorth(sig):
    var0 = np.var(sig)
    diff1 = np.diff(sig)
    var1 = np.var(diff1)
    mobility = np.sqrt(var1/var0) if var0>0 else 0.0
    diff2 = np.diff(diff1)
    var2 = np.var(diff2)
    complexity = (np.sqrt(var2/var1)/mobility) if var1>0 and mobility>0 else 0.0
    return var0, mobility, complexity

def band_power(sig, fs=1000):
    psd = np.abs(np.fft.rfft(sig))**2
    freqs = np.fft.rfftfreq(sig.size, 1/fs)
    bands = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),
             'beta':(13,30),'gamma':(30,100)}
    total = psd.sum()
    pw = {}
    for name,(lo,hi) in bands.items():
        idx = (freqs>=lo)&(freqs<hi)
        pw[name] = psd[idx].sum()/total if total>0 else 0.0
    return pw

def zero_crossing(sig):
    return ((sig[:-1]*sig[1:])<0).sum()/sig.size

# 4) assemble feature matrix
rows = []
for sig in signals:
    curvature = np.var(sig)
    ric_comp = np.mean(np.abs(np.diff(sig)))
    rec_lvl = np.std(sig)
    geom_var = np.mean(sig**2)
    temp_dyn = np.mean(np.diff(sig)**2)
    grav_mod = np.ptp(sig)
    topo = np.sum(sig>sig.mean())
    env = np.corrcoef(sig, np.arange(sig.size))[0,1]
    psd = np.abs(np.fft.rfft(sig))**2
    psd_norm = psd/psd.sum() if psd.sum()>0 else np.zeros_like(psd)
    ent = -np.sum(psd_norm*np.log2(psd_norm+1e-12))
    inter = curvature*ent

    act, mob, comp = hjorth(sig)
    pw = band_power(sig)
    zc = zero_crossing(sig)

    row = {
        'curvature': curvature,
        'ric_complexity': ric_comp,
        'recursion_level': rec_lvl,
        'geometry_var': geom_var,
        'temporal_dyn': temp_dyn,
        'gravitational_mod': grav_mod,
        'topo_changes': topo,
        'env_resp': env,
        'collapse_entropy': ent,
        'interaction_feat': inter,
        'hj_act': act,
        'hj_mobility': mob,
        'hj_complexity': comp,
        'zc_rate': zc
    }
    row.update({f"bp_{b}":v for b,v in pw.items()})
    rows.append(row)

df = pd.DataFrame(rows)
print("Feature matrix shape:", df.shape)
print(df.head(), "\n")

# 5) median‐split labels (guaranteed two classes)
df['label'] = (df['collapse_entropy'] > df['collapse_entropy'].median()).astype(int)
print("Labels distribution:\n", df['label'].value_counts(), "\n")

X = df.drop(columns='label').values
y = df['label'].values

# 6) pipelines + grids
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipelines = {
    'RF': Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('skb', SelectKBest(f_classif)),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'SVM': Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('skb', SelectKBest(f_classif)),
        ('clf', SVC(probability=True))
    ]),
    'GB': Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('skb', SelectKBest(f_classif)),
        ('clf', GradientBoostingClassifier())
    ]),
    'MLP': Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA()),
        ('skb', SelectKBest(f_classif)),
        ('clf', MLPClassifier(max_iter=500))
    ]),
}

param_grid = {
    'RF': {
        'pca__n_components':[10,15],
        'skb__k':[10, 'all'],
        'clf__n_estimators':[100,200],
        'clf__max_depth':[3,5]
    },
    'SVM': {
        'pca__n_components':[10,15],
        'skb__k':[10,'all'],
        'clf__C':[0.1,1],
        'clf__gamma':['scale','auto']
    },
    'GB': {
        'pca__n_components':[10,15],
        'skb__k':[10,'all'],
        'clf__n_estimators':[100,200],
        'clf__max_depth':[3,5]
    },
    'MLP': {
        'pca__n_components':[10,15],
        'skb__k':[10,'all'],
        'clf__hidden_layer_sizes':[(50,),(100,)],
        'clf__alpha':[1e-3,1e-4]
    }
}

best_models = {}
for name, pipe in pipelines.items():
    print(f"🔍 Tuning {name}...")
    grid = GridSearchCV(pipe, param_grid[name],
                        cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X, y)
    best_models[name] = grid
    print(f"  • {name} best AUC = {grid.best_score_:.2f}\n")

# 7) pick & display best
best_name, best_grid = max(best_models.items(),
                           key=lambda kv: kv[1].best_score_)
print(f"🏆 Best model: {best_name} (AUC={best_grid.best_score_:.2f})\n")

# 8) ROC for best
y_proba = cross_val_predict(best_grid.best_estimator_, X, y,
                            cv=cv, method='predict_proba')[:,1]
fpr,tpr,_ = roc_curve(y,y_proba)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,label=f"{best_name} AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("5-Fold Stratified ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 9) feature importances if tree‐based
if best_name in ('RF','GB'):
    imp = best_grid.best_estimator_.named_steps['clf'].feature_importances_
    names = df.drop(columns='label').columns
    plt.figure(figsize=(8,4))
    plt.bar(names, imp)
    plt.xticks(rotation=45,ha='right')
    plt.title(f"{best_name} Feature Importances")
    plt.tight_layout()
    plt.show()

print("✅ All done.")
