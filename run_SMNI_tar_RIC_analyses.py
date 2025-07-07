#!/usr/bin/env python3
# ───────────────────────────────────────────────────────────
# run_SMNI_tar_RIC_analyses.py
# End-to-end pipeline: RIC metrics + four companion analyses
# on SMNI tar archives that contain *.rd.gz EEG chunks.
# Tested with Python ≥3.9, MNE ≥1.6, NumPy ≥1.22
# ───────────────────────────────────────────────────────────

import os, glob, tarfile, io, gzip, warnings
import numpy  as np
import pandas as pd
from scipy.signal import welch
from scipy.stats  import entropy, ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.cluster  import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt; import seaborn as sns
import mne, umap

warnings.filterwarnings("ignore")
sns.set(context="paper", style="whitegrid")
rng = np.random.default_rng(42)

# ───────── configuration ──────────────────────────────────
DATA_ROOT = r"C:\Users\mahsa\OneDrive\Desktop\RIC_EEG\SMNI_tars"  # ← your folder
FS   = 256                       # SMNI sampling rate
WIN  = FS                        # 1-second window for RIC
BINS = 6                         # symbol bins
ALPHA = BETA = 1.0               # RIC coefficients
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),
         "beta":(13,30),"gamma":(30,45)}

TXT_EXT = (".txt", ".dat", ".smni", ".csv", ".asc", ".tsv",
           ".rd", ".rd.gz", ".gz")            # ← includes plain .gz
# ───────────────────────────────────────────────────────────

def ric_k(sig):
    """Return mean RIC curvature (K), dummy std and collapse flag."""
    n = len(sig)//WIN
    if n < 2:
        return np.nan, np.nan, np.nan
    blk = sig[:n*WIN].reshape(n, WIN).mean(1)
    q   = np.quantile(blk, np.linspace(0, 1, BINS+1))
    lab = np.digitize(blk, q[1:-1])
    p   = np.bincount(lab, minlength=BINS) / len(lab)
    S   = entropy(p, base=np.e)
    R   = np.corrcoef(lab[:-1], lab[1:])[0, 1]
    K   = ALPHA * R - BETA * S
    return K, 0.0, float(K < 0)

def band_power(sig):
    f, P = welch(sig, FS, nperseg=FS*4)
    return {bn: P[(f >= lo) & (f < hi)].mean() for bn, (lo, hi) in BANDS.items()}

# ───────── loader for rd.gz chunks + ASCII fallback ───────
def raw_from_bytes(byt: bytes, name: str):
    """Return an MNE Raw object from bytes of rd(.gz) or ASCII file."""
    ext = os.path.splitext(name)[1].lower()

    # SMNI rd or rd.gz binary chunk
    if ext in (".rd", ".gz"):
        if ext == ".gz":
            byt = gzip.decompress(byt)
        sig = np.frombuffer(byt, dtype="<i2").astype(np.float32) * 1e-6  # µV→V
        return mne.io.RawArray(
            sig[np.newaxis, :],
            mne.create_info(["eeg"], FS, "eeg"),
            verbose=False,
        )

    # Plain ASCII fallback
    clean = b"\n".join(
        line for line in byt.splitlines() if not line.lstrip().startswith(b"#")
    )
    arr = np.loadtxt(io.BytesIO(clean))
    data = arr.mean(1) if arr.ndim > 1 else arr
    return mne.io.RawArray(
        data[np.newaxis, :],
        mne.create_info(["eeg"], FS, "eeg"),
        verbose=False,
    )

# ───────── find archives ──────────────────────────────────
tar_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "co2[a,c]*.tar*")))
print(f"Found {len(tar_paths)} tar archives under {DATA_ROOT}")
if not tar_paths:
    raise SystemExit("✘  No archives found — check DATA_ROOT path.")

SUBJ, EPOCH = [], []

# ───────── iterate archives ───────────────────────────────
for tar_path in tar_paths:
    with tarfile.open(tar_path, "r:*") as T:
        members = [
            m for m in T.getmembers()
            if os.path.splitext(m.name)[1].lower() in TXT_EXT
        ]
        if not members:
            print("⚠ No EEG chunks in", os.path.basename(tar_path))
            continue

        sigs = []
        for m in members:
            try:
                raw = raw_from_bytes(T.extractfile(m).read(), m.name)
                sigs.append(raw.get_data()[0])
            except Exception as e:
                print("  skip", m.name, "→", e)

        if not sigs:
            continue

        # align lengths: trim every chunk to the shortest
        min_len = min(len(s) for s in sigs)
        sigs = [s[:min_len] for s in sigs]
        gfp  = np.vstack(sigs).mean(0)

        meanK, _, col = ric_k(gfp)
        SUBJ.append(
            {
                "file": os.path.basename(tar_path),
                "meanK": meanK,
                "collapse": col,
                **band_power(gfp),
                "gfp": gfp,
            }
        )

        # store 2-s epochs for cluster analysis
        ep_raw = mne.io.RawArray(
            gfp[np.newaxis, :],
            mne.create_info(["gfp"], FS, "eeg"),
            verbose=False,
        )
        for ep in mne.make_fixed_length_epochs(ep_raw, 2, preload=True).get_data().squeeze():
            EPOCH.append({"file": os.path.basename(tar_path),
                          "K": ric_k(ep)[0],
                          **band_power(ep)})

# stop if still empty
if not SUBJ:
    raise SystemExit("✘  Still no usable EEG found – loader tweak needed.")

# ───────── DataFrames ready ───────────────────────────────
subj_df  = pd.DataFrame(SUBJ).set_index("file")
epoch_df = pd.DataFrame(EPOCH)

# ───────── A  violin of mean K ─────────────────────────────
plt.figure(figsize=(9, 4))
sns.violinplot(data=subj_df["meanK"], orient="h", inner="point", cut=0)
plt.title("Mean recursive curvature (K) — SMNI set")
plt.xlabel("Mean K")
plt.tight_layout()
plt.savefig("A_SMNI_tar_violin.png", dpi=300)
subj_df.drop(columns="gfp").to_csv("A_SMNI_tar_stats.csv")

# ───────── B  UMAP + DBSCAN on epochs ─────────────────────
X   = epoch_df[["K"] + list(BANDS.keys())].fillna(0)
emb = umap.UMAP(random_state=42).fit_transform(StandardScaler().fit_transform(X))
cluster = DBSCAN(eps=0.8, min_samples=10).fit_predict(emb)
epoch_df["cluster"] = cluster
plt.figure(figsize=(6, 5))
sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=cluster, s=6,
                palette="tab10", legend=False)
plt.title("UMAP of 2-s epochs  (RIC + power)")
plt.tight_layout()
plt.savefig("B_SMNI_tar_umap.png", dpi=300)
epoch_df.to_csv("B_SMNI_tar_clusters.csv", index=False)

# ───────── C  surrogate (phase-random) test on mean K ─────
realK = [d["meanK"] for d in SUBJ]
surK  = [ric_k(np.random.permutation(d["gfp"]))[0] for d in SUBJ]
t, p  = ttest_rel(realK, surK)
plt.figure(figsize=(4, 4))
sns.boxplot(data=pd.DataFrame({"Real": realK, "Surrogate": surK}))
plt.title(f"Phase-random surrogate test  p={p:.1e}")
plt.tight_layout()
plt.savefig("C_SMNI_tar_surrogate.png", dpi=300)
with open("C_SMNI_tar_stats.txt", "w") as f:
    f.write(f"paired t={t:.3f}, p={p:.3e}\n")

# ───────── D  burst-injection ROC (RIC vs α-power) ────────
def burst(sig):
    idx = rng.integers(0, len(sig) - FS // 2)
    out = sig.copy()
    out[idx : idx + FS // 2] += 1.5 * np.sin(2 * np.pi * 12 * np.arange(FS // 2) / FS)
    return out

y, ks, ap = [], [], []
for d in SUBJ:
    sig, baseK = d["gfp"], d["meanK"]
    for _ in range(30):
        y   += [0, 1]
        sigb = burst(sig)
        ks  += [baseK, ric_k(sigb)[0]]
        ap  += [band_power(sig)["alpha"],
                band_power(sigb)["alpha"]]

auc_k = roc_auc_score(y, -np.array(ks))
auc_a = roc_auc_score(y,  np.array(ap))
plt.figure(figsize=(5, 5))
RocCurveDisplay.from_predictions(y, -np.array(ks), name=f"RIC  (AUC {auc_k:.2f})")
RocCurveDisplay.from_predictions(y,  np.array(ap), name=f"α-power (AUC {auc_a:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.tight_layout()
plt.savefig("D_SMNI_tar_ROC.png", dpi=300)
with open("D_SMNI_tar_auc.txt", "w") as f:
    f.write(f"AUC_RIC {auc_k:.3f}   AUC_alpha {auc_a:.3f}\n")

# ───────── E  Isolation-Forest outlier scan ───────────────
feat_full = subj_df[["meanK", "collapse"] + list(BANDS.keys())]
feat_spec = subj_df[list(BANDS.keys())]
iso_f = IsolationForest(n_estimators=300, contamination=0.12,
                        random_state=42).fit(feat_full)
iso_s = IsolationForest(n_estimators=300, contamination=0.12,
                        random_state=42).fit(feat_spec)
subj_df["score_fus"] = -iso_f.decision_function(feat_full)
subj_df["score_spc"] = -iso_s.decision_function(feat_spec)
subj_df.drop(columns="gfp").reset_index()[
    ["file", "score_fus", "score_spc"]
].to_csv("E_SMNI_tar_outliers.csv", index=False)
plt.figure(figsize=(6, 4))
sns.scatterplot(data=subj_df, x="score_spc", y="score_fus", s=20)
q_spc = np.percentile(subj_df["score_spc"], 88)
q_fus = np.percentile(subj_df["score_fus"], 88)
plt.axvline(q_spc, ls="--", c="r"); plt.axhline(q_fus, ls="--", c="r")
plt.xlabel("Outlier score (spectral)"); plt.ylabel("Outlier score (fusion)")
plt.title("Isolation-Forest anomaly scan")
plt.tight_layout(); plt.savefig("E_SMNI_tar_IForest.png", dpi=300)

print("\n✔  Five analyses on SMNI tar set finished — outputs saved.")
