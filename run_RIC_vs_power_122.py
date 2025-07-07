#!/usr/bin/env python3
# run_RIC_vs_power_122.py  •  5 analyses on 122 EEG recordings
# ============================================================

# ── USER: point this to the folder that contains the EEG files ─────────────
DATA_ROOT = r"C:\Users\mahsa\OneDrive\Desktop\RIC_EEG\data122"
#  (absolute or relative path; sub-folders are scanned automatically)
# --------------------------------------------------------------------------

import os, glob, warnings, numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats  import entropy, ttest_rel
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import DBSCAN
from sklearn.ensemble      import IsolationForest
import matplotlib.pyplot as plt, seaborn as sns
import mne, pymatreader

# optional dependency
try:
    import umap
except ImportError:
    raise SystemExit("✘  Please install UMAP:  pip install umap-learn")

warnings.filterwarnings("ignore")
sns.set(context="paper", style="whitegrid")
rng = np.random.default_rng(42)

# ── helper: RIC, power, surrogate, loader ──────────────────
FS   = 250; WIN = FS; BINS = 6; ALPHA = BETA = 1.0
BANDS={"delta":(1,4),"theta":(4,8),"alpha":(8,13),
       "beta":(13,30),"gamma":(30,45)}
def ric_from_signal(sig):
    n = len(sig)//WIN
    if n<2: return np.nan,np.nan,np.nan
    m = sig[:n*WIN].reshape(n,WIN).mean(1)
    q = np.quantile(m, np.linspace(0,1,BINS+1))
    lab = np.digitize(m, q[1:-1])
    p   = np.bincount(lab,minlength=BINS)/len(lab)
    S   = entropy(p, base=np.e)
    R   = np.corrcoef(lab[:-1],lab[1:])[0,1]
    K   = ALPHA*R - BETA*S
    return K,0.0,float(K<0)
def band_power(sig):
    f,P = welch(sig, FS, nperseg=FS*4)
    return {bn:P[(f>=lo)&(f<hi)].mean() for bn,(lo,hi) in BANDS.items()}
def phase_random(sig):
    fft = np.fft.rfft(sig)
    fft[1:] *= np.exp(1j*rng.uniform(0,2*np.pi,len(fft)-1))
    return np.fft.irfft(fft, n=len(sig))
def smart_loader(fname, fs_target=FS):
    if fname.lower().endswith(".edf"):
        raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
    elif fname.lower().endswith(".set"):
        raw = mne.io.read_raw_eeglab(fname, preload=True, verbose=False)
    elif fname.lower().endswith(".mat"):
        mat  = pymatreader.read_mat(fname)
        arr  = np.asarray(next(iter(mat.values()))).squeeze()
        data = arr.mean(0) if arr.ndim>1 else arr
        raw  = mne.io.RawArray(
               data[np.newaxis,:],
               mne.create_info(["eeg"], fs_target, "eeg"), verbose=False)
    else:
        raise ValueError("unknown format")
    raw.resample(fs_target, npad="auto")
    return raw

# ── discover files ---------------------------------------------------------
EEG_EXT = (".edf",".set",".mat")
FILES   = [p for ext in EEG_EXT
           for p in glob.glob(os.path.join(DATA_ROOT,"**","*"+ext),
                              recursive=True)]
print(f"Found {len(FILES)} EEG files under {DATA_ROOT}")

if not FILES:
    raise SystemExit("✘  No .edf/.set/.mat files found — check DATA_ROOT.")

# ── feature extraction -----------------------------------------------------
SUBJ, EPOCH, good = [], [], []
for f in FILES:
    try:
        raw = smart_loader(f)
    except Exception as e:
        print("skip", os.path.basename(f), "→", e); continue
    good.append(f)
    gfp = raw.get_data(picks="eeg").mean(0)
    K,_,c = ric_from_signal(gfp)
    SUBJ.append({"file":f,"meanK":K,"collapse":c,**band_power(gfp)})
    for ep in mne.make_fixed_length_epochs(raw,2,preload=True)\
                 .get_data(picks="eeg").mean(1):
        K_ep,_,_ = ric_from_signal(ep)
        EPOCH.append({"file":f,"K":K_ep,**band_power(ep)})

subj_df  = pd.DataFrame(SUBJ).set_index("file")
epoch_df = pd.DataFrame(EPOCH)

# ── Analysis A: curvature violin ------------------------------------------
plt.figure(figsize=(9,4))
sns.violinplot(data=subj_df["meanK"],orient="h",inner="point",cut=0)
plt.title("Mean recursive curvature (K) — all subjects")
plt.xlabel("Mean K"); plt.tight_layout()
plt.savefig("A_fingerprint_violin.png", dpi=300)
subj_df.to_csv("A_subject_stats.csv")

# ── Analysis B: UMAP + DBSCAN ---------------------------------------------
import umap
X  = epoch_df[["K"]+list(BANDS.keys())].fillna(0)
emb = umap.UMAP(random_state=42).fit_transform(StandardScaler().fit_transform(X))
cl  = DBSCAN(eps=0.8, min_samples=10).fit_predict(emb)
epoch_df["cluster"]=cl
plt.figure(figsize=(6,5))
sns.scatterplot(x=emb[:,0],y=emb[:,1],hue=cl,s=6,legend=False,palette="tab10")
plt.title("UMAP of 2-s epochs  (RIC + power)")
plt.tight_layout(); plt.savefig("B_umap_clusters.png", dpi=300)
epoch_df.to_csv("B_epoch_clusters.csv",index=False)

# ── Analysis C: surrogate K test ------------------------------------------
realK,surK = [],[]
for f in good:
    sig = smart_loader(f).get_data(picks="eeg").mean(0)
    realK.append(ric_from_signal(sig)[0])
    surK.append(ric_from_signal(phase_random(sig))[0])
t,p = ttest_rel(realK,surK)
plt.figure(figsize=(4,4))
sns.boxplot(data=pd.DataFrame({"Real":realK,"Surrogate":surK}))
plt.title(f"Phase-random surrogate test  p={p:.1e}")
plt.tight_layout(); plt.savefig("C_surrogate_K.png", dpi=300)
with open("C_surrogate_stats.txt","w") as w:
    w.write(f"paired t = {t:.3f}, p = {p:.3e}\n")

# ── Analysis D: burst ROC ---------------------------------------------------
def inject_burst(sig):
    idx=rng.integers(0,len(sig)-FS//2)
    sig2=sig.copy()
    sig2[idx:idx+FS//2]+=1.5*np.sin(2*np.pi*12*np.arange(FS//2)/FS)
    return sig2
y,ks,ap = [],[],[]
for f in good:
    sig = smart_loader(f).get_data(picks="eeg").mean(0)
    baseK, baseA = ric_from_signal(sig)[0], band_power(sig)["alpha"]
    for _ in range(40):
        y+=[0,1]
        sigb=inject_burst(sig)
        ks += [baseK, ric_from_signal(sigb)[0]]
        ap += [baseA, band_power(sigb)["alpha"]]
auc_k = roc_auc_score(y,-np.array(ks))
auc_a = roc_auc_score(y, np.array(ap))
plt.figure(figsize=(5,5))
RocCurveDisplay.from_predictions(y,-np.array(ks),name=f"RIC {auc_k:.2f}")
RocCurveDisplay.from_predictions(y, np.array(ap),name=f"α-power {auc_a:.2f}")
plt.plot([0,1],[0,1],'k--'); plt.tight_layout()
plt.savefig("D_burst_ROC.png", dpi=300)
with open("D_burst_auc.txt","w") as w:
    w.write(f"AUC_RIC {auc_k:.3f}  AUC_alpha {auc_a:.3f}\n")

# ── Analysis E: Isolation-Forest ------------------------------------------
feat_full = subj_df[["meanK","collapse"]+list(BANDS.keys())]
feat_spec = subj_df[list(BANDS.keys())]
ISO_f = IsolationForest(300, contamination=0.12, random_state=42).fit(feat_full)
ISO_s = IsolationForest(300, contamination=0.12, random_state=42).fit(feat_spec)
subj_df["score_fus"] = -ISO_f.decision_function(feat_full)
subj_df["score_spc"] = -ISO_s.decision_function(feat_spec)
subj_df.reset_index()[["file","score_fus","score_spc"]]\
       .to_csv("E_isolation_outliers.csv", index=False)
plt.figure(figsize=(6,4))
sns.scatterplot(data=subj_df,x="score_spc",y="score_fus",s=20)
plt.axvline(np.percentile(subj_df["score_spc"],88),ls='--',c='r')
plt.axhline(np.percentile(subj_df["score_fus"],88),ls='--',c='r')
plt.xlabel("Outlier score (spectral)"); plt.ylabel("Outlier score (fusion)")
plt.title("Isolation-Forest anomaly scan"); plt.tight_layout()
plt.savefig("E_isolation_scores.png", dpi=300)

print("\n✔ All five analyses finished — outputs saved.")
