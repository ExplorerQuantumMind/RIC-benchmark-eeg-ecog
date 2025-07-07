#!/usr/bin/env python3
# run_all_RIC_analyses.py  –  five unsupervised RIC-vs-EEG analyses
# =================================================================

import os, warnings, numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats  import entropy, ttest_rel
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.ensemble      import RandomForestClassifier, IsolationForest
from sklearn.metrics       import RocCurveDisplay
from sklearn.cluster       import DBSCAN
import umap
import matplotlib.pyplot as plt, seaborn as sns
import mne; warnings.filterwarnings("ignore")
sns.set(context="paper", style="whitegrid")
rng = np.random.default_rng(42)

# ───── Helper: phase-random surrogate ───────────────────────
def phase_random(sig):
    fft = np.fft.rfft(sig)
    angles = np.random.uniform(0, 2*np.pi, len(fft)-1)
    fft[1:] *= np.exp(1j * angles)
    return np.fft.irfft(fft, n=len(sig))

# ───── Build file list (1…82) – sub-083 excluded ────────────
FILES = [f"sub-{i:03d}_task-eyesclosed_eeg.set" for i in range(1, 83)]
FILES = [f for f in FILES if os.path.exists(f)]
print(f"Found {len(FILES)} valid .set files (sub-083 excluded).")

# ───── Parameters & helpers ─────────────────────────────────
FS, WIN = 250, 250          # resample Hz, window samples (1 s)
BINS, ALPHA, BETA = 6, 1.0, 1.0
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),
         "beta":(13,30),"gamma":(30,45)}

def ric_from_signal(sig):
    n = len(sig)//WIN
    if n < 2:
        return np.nan, np.nan, np.nan
    means = sig[:n*WIN].reshape(n,WIN).mean(1)
    q = np.quantile(means, np.linspace(0,1,BINS+1))
    lab = np.digitize(means, q[1:-1])
    p = np.bincount(lab, minlength=BINS)/len(lab)
    S = entropy(p, base=np.e)
    R = np.corrcoef(lab[:-1], lab[1:])[0,1]
    K = ALPHA*R - BETA*S
    return K, 0.0, float(K<0)

def band_power(sig):
    f,Pxx = welch(sig, fs=FS, nperseg=FS*4)
    return {bn:Pxx[(f>=lo)&(f<hi)].mean() for bn,(lo,hi) in BANDS.items()}

# ───── Load data & extract features (skip bad files) ────────
SUBJ, EPOCH, good_files = [], [], []
for f in FILES:
    try:
        raw = mne.io.read_raw_eeglab(f, preload=False, verbose=False)
    except Exception as err:
        print(f"⚠️  Skipping {f} — {err}")
        continue

    good_files.append(f)
    raw.resample(FS, npad="auto")
    gfp = raw.get_data(picks="eeg").mean(0)

    K, _, col = ric_from_signal(gfp)
    SUBJ.append({"file":f, "meanK":K, "collapse":col, **band_power(gfp)})

    epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True)
    for ep in epochs.get_data(picks="eeg").mean(1):
        K_ep,_,_ = ric_from_signal(ep)
        EPOCH.append({"file":f,"K":K_ep, **band_power(ep)})

subj_df  = pd.DataFrame(SUBJ).set_index("file")
epoch_df = pd.DataFrame(EPOCH)

# ───── Analysis 1 – curvature fingerprints ──────────────────
plt.figure(figsize=(9,4))
sns.violinplot(data=subj_df["meanK"], inner="point", orient="h", cut=0)
plt.title("Mean recursive curvature (K) — all subjects")
plt.xlabel("Mean K"); plt.tight_layout()
plt.savefig("fingerprint_violin.png", dpi=300)
subj_df.to_csv("subject_K_stats.csv")

# ───── Analysis 2 – UMAP + DBSCAN clusters ──────────────────
um = umap.UMAP(random_state=42)
X2 = epoch_df[["K"]+list(BANDS.keys())].fillna(0).to_numpy()
emb = um.fit_transform(StandardScaler().fit_transform(X2))
cl  = DBSCAN(eps=0.8, min_samples=10).fit_predict(emb)
epoch_df["cluster"] = cl
plt.figure(figsize=(6,5))
sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=cl, palette="tab10", s=8, legend=False)
plt.title("UMAP of 2-s epochs  (RIC + power)")
plt.tight_layout(); plt.savefig("umap_clusters.png", dpi=300)
epoch_df.to_csv("epoch_clusters.csv", index=False)

# ───── Analysis 3 – surrogate test ──────────────────────────
real_K, sur_K = [], []
for f in good_files:
    sig = mne.io.read_raw_eeglab(f, preload=True, verbose=False
             ).resample(FS, npad="auto").get_data(picks="eeg").mean(0)
    real_K.append(ric_from_signal(sig)[0])
    sur_K.append(ric_from_signal(phase_random(sig))[0])
t,p = ttest_rel(real_K, sur_K)
plt.figure(figsize=(4,4))
sns.boxplot(data=pd.DataFrame({"Real":real_K,"Surrogate":sur_K}))
plt.title(f"Phase-random surrogate test   p={p:.1e}")
plt.tight_layout(); plt.savefig("surrogate_K_comparison.png", dpi=300)
with open("surrogate_stats.txt","w") as w:
    w.write(f"paired t  Real vs Surrogate:  t={t:.3f}  p={p:.3e}\n")

# ───── Analysis 4 – perturb-and-recover ROC ────────────────
def inject_burst(sig):
    idx = rng.integers(0, len(sig)-FS//2)
    burst = 1.5*np.sin(2*np.pi*12*np.arange(FS//2)/FS)
    sig2 = sig.copy(); sig2[idx:idx+len(burst)] += burst
    return sig2

y_true, ric_scr, alp_scr = [], [], []
for f in good_files:
    sig = mne.io.read_raw_eeglab(f, preload=True, verbose=False
            ).resample(FS, npad="auto").get_data(picks="eeg").mean(0)
    baseK   = ric_from_signal(sig)[0]
    baseAlp = band_power(sig)["alpha"]
    for _ in range(60):
        y_true += [0,1]
        sig_b  = inject_burst(sig)
        ric_scr += [baseK, ric_from_signal(sig_b)[0]]
        alp_scr += [baseAlp, band_power(sig_b)["alpha"]]

auc_R = roc_auc_score(y_true, -np.array(ric_scr))
auc_A = roc_auc_score(y_true,  np.array(alp_scr))
fig, ax = plt.subplots(figsize=(5,5))
RocCurveDisplay.from_predictions(y_true, -np.array(ric_scr),
                                 name=f"RIC (AUC {auc_R:.2f})", ax=ax)
RocCurveDisplay.from_predictions(y_true, alp_scr,
                                 name=f"α-power (AUC {auc_A:.2f})", ax=ax)
ax.plot([0,1],[0,1],"k--"); ax.set_title("Burst-detection ROC")
plt.tight_layout(); plt.savefig("perturb_roc.png", dpi=300)
with open("perturb_auc.txt","w") as w:
    w.write(f"RIC   AUC {auc_R:.3f}\nalpha AUC {auc_A:.3f}\n")

# ───── Analysis 5 – Isolation-Forest outliers ───────────────
feat_full = subj_df[["meanK","collapse"]+list(BANDS.keys())]
feat_spec = subj_df[list(BANDS.keys())]
ISO_f = IsolationForest(n_estimators=300, contamination=0.12,
                        random_state=42).fit(feat_full)
ISO_s = IsolationForest(n_estimators=300, contamination=0.12,
                        random_state=42).fit(feat_spec)
subj_df["score_fus"] = -ISO_f.decision_function(feat_full)
subj_df["score_spc"] = -ISO_s.decision_function(feat_spec)
subj_df.reset_index()[["file","score_fus","score_spc"]]\
       .to_csv("isolation_outliers.csv", index=False)

plt.figure(figsize=(6,4))
sns.scatterplot(data=subj_df, x="score_spc", y="score_fus")
plt.xlabel("Outlier score (spectral)"); plt.ylabel("Outlier score (fusion)")
plt.axvline(np.percentile(subj_df["score_spc"], 88), ls="--", c="r")
plt.axhline(np.percentile(subj_df["score_fus"], 88), ls="--", c="r")
plt.title("Isolation-Forest anomaly scan")
plt.tight_layout(); plt.savefig("isolation_scores.png", dpi=300)

print("\nAll five analyses completed — outputs saved!")
