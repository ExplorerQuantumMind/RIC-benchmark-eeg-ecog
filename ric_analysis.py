"""
Recursive Informational Curvature (RIC) analysis – chunk-safe version
--------------------------------------------------------------------
• Reads each .set EEG file in 60-second chunks (memory-friendly)
• Computes symbolic entropy S, recursive gain R, and curvature K = α·R – β·S
• Saves a CSV with mean/SD/collapse-rate and a PNG with bar plots
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
import mne

# ------------------- USER-ADJUSTABLE PARAMETERS --------------------
window_size      = 500   # samples per symbolic window  (500 ≈ 1 s at 500 Hz)
n_bins           = 6     # bins for discretisation
alpha, beta      = 1.0, 1.0          # RIC coefficients  K = α·R – β·S
chunk_duration   = 60                # seconds per chunk loaded into RAM
eeg_file_paths   = [                 # update list if filenames differ
    "sub-001_task-eyesclosed_eeg.set",
    "sub-002_task-eyesclosed_eeg.set",
    "sub-003_task-eyesclosed_eeg.set",
    "sub-004_task-eyesclosed_eeg.set",
    "sub-005_task-eyesclosed_eeg.set",
    "sub-006_task-eyesclosed_eeg.set",
    "sub-007_task-eyesclosed_eeg.set",
    "sub-008_task-eyesclosed_eeg.set",
    "sub-009_task-eyesclosed_eeg.set",
]
# -------------------------------------------------------------------


def process_subject(path: str) -> dict:
    reader = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
    sfreq           = reader.info["sfreq"]
    total_seconds   = reader.n_times / sfreq          # exact duration
    n_channels      = len(reader.ch_names)

    curvature_vals  = []

    # iterate in 60-s steps but never exceed recording end
    start = 0.0
    while start < total_seconds:
        stop = min(start + chunk_duration, total_seconds - 1/sfreq)  # safe tmax
        raw_chunk = (
            reader.copy()
            .crop(tmin=start, tmax=stop, include_tmax=False)
            .load_data()
        )
        data = raw_chunk.get_data()                # shape (channels, samples)
        n_windows = data.shape[1] // window_size
        if n_windows == 0:         # chunk shorter than one window – skip
            start += chunk_duration
            continue

        symbolic = np.zeros((n_channels, n_windows), dtype=int)
        for ch in range(n_channels):
            seg = (
                data[ch, : n_windows * window_size]
                .reshape(n_windows, window_size)
                .mean(axis=1)
                .reshape(-1, 1)
            )
            disc = KBinsDiscretizer(n_bins=n_bins,
                                    encode="ordinal",
                                    strategy="quantile")
            symbolic[ch] = disc.fit_transform(seg).flatten()

        # entropy, recursive gain, curvature
        for ch in range(n_channels):
            sym = symbolic[ch]
            if sym.size < 2:
                continue
            p = np.bincount(sym, minlength=n_bins) / sym.size
            S = entropy(p, base=np.e)
            R = np.corrcoef(sym[:-1], sym[1:])[0, 1]
            K = alpha * R - beta * S
            curvature_vals.append(K)

        start += chunk_duration    # next chunk

    curvature_vals = np.array(curvature_vals)
    return {
        "Subject": os.path.basename(path),
        "Mean RIC Curvature": np.nanmean(curvature_vals),
        "Std  RIC Curvature": np.nanstd(curvature_vals),
        "Collapse Rate": np.mean(curvature_vals < 0.0),
    }


# ------------------------- RUN ANALYSIS ----------------------------
results = [process_subject(fp) for fp in eeg_file_paths]
df = pd.DataFrame(results).sort_values("Subject")
df.to_csv("ric_results.csv", index=False)
print("✔ Saved  ric_results.csv")

# ------------------------- PLOTTING --------------------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

sns.barplot(data=df, x="Subject", y="Mean RIC Curvature",
            ax=axes[0], palette="Blues_d")
axes[0].set_title("Mean RIC Curvature")
axes[0].tick_params(axis="x", rotation=45)

sns.barplot(data=df, x="Subject", y="Std  RIC Curvature",
            ax=axes[1], palette="Greens_d")
axes[1].set_title("Curvature Standard Deviation")
axes[1].tick_params(axis="x", rotation=45)

sns.barplot(data=df, x="Subject", y="Collapse Rate",
            ax=axes[2], palette="Reds_d")
axes[2].set_title("Collapse Rate (proportion of K < 0)")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("ric_analysis_plots.png")
plt.close()
print("✔ Saved  ric_analysis_plots.png")
