import os
import glob
import scipy.io
import h5py
import numpy as np
import pandas as pd

# Directory containing your .mat files
DATA_DIR = '/mnt/data/'

# Helper to load MATLAB v7.3 HDF5 files (if needed), else fallback to scipy.loadmat
def load_mat(path):
    try:
        # Try HDF5-based loading
        with h5py.File(path, 'r') as f:
            # Return dict of all datasets at root
            return {k: np.array(v).squeeze() for k, v in f.items()}
    except (OSError, KeyError):
        # Fall back on scipy for older .mat versions
        return {k: v.squeeze() for k, v in scipy.io.loadmat(path).items() if not k.startswith('__')}

# 1) Gather file paths
lfp_files = sorted(glob.glob(os.path.join(DATA_DIR, 'LFP_ch*.mat')))
other_files = {
    'task_info': os.path.join(DATA_DIR, 'Task_info.mat'),
    'movie_start': os.path.join(DATA_DIR, 'Movie_start_time.mat'),
    'electrode_outline': os.path.join(DATA_DIR, 'Electrode_fmri_outline_with_MW.mat'),
}

# 2) Load LFP channels into a (n_ch x T) array
lfp_list = []
for fpath in lfp_files:
    data = load_mat(fpath)
    # Assume the variable name is the same as file basename without extension
    varname = os.path.splitext(os.path.basename(fpath))[0]
    sig = data.get(varname)
    if sig is None:
        # If unknown, just pick the first array present
        sig = next(iter(data.values()))
    lfp_list.append(sig.astype(float))
lfp = np.vstack(lfp_list)  # shape (5, timepoints)

# 3) Compute basic “RIC” features per channel
def compute_basic_ric_features(lfp_mat):
    feats = {
        'mean_power': np.mean(lfp_mat**2, axis=1),
        'variance': np.var(lfp_mat, axis=1),
        'spectral_entropy': []
    }
    for ch in range(lfp_mat.shape[0]):
        sig = lfp_mat[ch]
        psd = np.abs(np.fft.rfft(sig))**2
        psd /= psd.sum()
        sh_entropy = -np.sum(psd * np.log2(psd + 1e-12))
        feats['spectral_entropy'].append(sh_entropy)
    feats['spectral_entropy'] = np.array(feats['spectral_entropy'])
    return feats

ric_features = compute_basic_ric_features(lfp)

# 4) Build a summary DataFrame
df = pd.DataFrame({
    'channel': [f'ch{i+1}' for i in range(lfp.shape[0])],
    'mean_power': ric_features['mean_power'],
    'variance': ric_features['variance'],
    'spectral_entropy': ric_features['spectral_entropy']
})

# 5) Save and print
out_csv = os.path.join(DATA_DIR, 'new_dataset_basic_ric_features.csv')
df.to_csv(out_csv, index=False)
print(f"RIC summary saved to {out_csv}\n")
print(df)
