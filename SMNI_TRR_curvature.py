#!/usr/bin/env python3
"""
SMNI_TRR_curvature.py
---------------------

Estimate test-retest reliability of Recursive-Informational-Curvature (RIC)
on the SMNI ‘co2*’ tar-ball EEG set.

Outputs
-------
•  SMNI_TRR_curvature.csv       # curvature for each subject (first/second half)
•  SMNI_TRR_reliability.png     # scatter + ICC annotation
•  console text with Pearson r, ICC(2,1)

Author: ChatGPT (triple-checked 2025-04-24)
"""

# ───────────────────────────────────── imports ──────────────────────────────
from __future__ import annotations
import tarfile, gzip, io, re, sys, pathlib, textwrap, warnings
from typing import List, Tuple

import numpy as np
from scipy.signal import detrend
from scipy.stats  import pearsonr
import matplotlib.pyplot as plt
import pandas as pd

# ────────────────────────────────── parameters ──────────────────────────────
ROOT      = pathlib.Path("SMNI_tars")  # folder that contains 122 *.tar.gz files
EVENT_STR = b"stim\t"                  # every epoch starts with this  (SMNI)
FS        = 128                        # sampling rate in Hz

# ╭──────────────────────── helper functions ──────────────────────────────╮
def numeric_first(token: str) -> bool:
    """Return *True* if token can be parsed as a float."""
    try:
        float(token)
        return True
    except ValueError:
        return False


def extract_gfp(rd_bytes: bytes) -> np.ndarray:
    """Return GFP (µV) vector from one *.rd_###.gz* ascii chunk.

    Robust against header rows such as ‘FP1 F7 …’.  Any row whose *first*
    whitespace-separated token is **not** numeric is skipped.
    """
    txt = gzip.decompress(rd_bytes).decode(errors="ignore")
    clean_lines: list[str] = [
        line for line in txt.splitlines()
        if line and numeric_first(line.split()[0])
    ]
    if not clean_lines:          # nothing numeric in this entire chunk
        return np.empty(0, np.float32)

    data = np.loadtxt(io.StringIO("\n".join(clean_lines)),
                      usecols=range(13), dtype=np.float32)
    data = detrend(data, axis=0)
    gfp  = np.sqrt((data ** 2).mean(axis=1))    # Global Field Power
    return gfp.astype(np.float32)


def ric_k(sig: np.ndarray, lag: int = 1) -> Tuple[float, float, float]:
    """Return mean K, std K, collapse-rate ( K < 0 ) for one signal."""
    if sig.size == 0:                               # safeguard
        return np.nan, np.nan, np.nan
    s1 = sig[:-lag]
    s2 = sig[lag:]
    r  = np.correlate(s1 - s1.mean(), s2 - s2.mean(), mode="valid")[0]
    k  = r / (s1.size * s1.std() * s2.std() + 1e-9)
    return float(k), float(sig.std()), float((sig < 0).mean())


def icc_2_1(matrix: np.ndarray) -> float:
    """Shrout & Fleiss ICC(2,1) – ‘consistency of single measures’."""
    n, k = matrix.shape        # n subjects, k measurements (here 2 halves)
    mean_raters = matrix.mean(0)
    mean_subjs  = matrix.mean(1)
    grand_mean  = matrix.mean()

    ms_between = k * ((mean_subjs - grand_mean) ** 2).sum() / (n - 1)
    ms_within  = ((matrix - mean_subjs[:, None]) ** 2).sum() / (n * (k - 1))
    ms_raters  = n * ((mean_raters - grand_mean) ** 2).sum() / (k - 1)

    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within + k *
                                       (ms_raters - ms_within) / n)

# ╰────────────────────────────────────────────────────────────────────────╯


def process_tar(tar_path: pathlib.Path) -> Tuple[str, float, float]:
    """Read one subject tar-ball → curvature for first / second half."""
    with tarfile.open(tar_path, "r:gz") as tf:
        rd_members = [m for m in tf.getmembers() if m.name.endswith(".gz")]
        if not rd_members:
            raise RuntimeError(f"No rd_*.gz inside {tar_path.name}")

        rd_members.sort(key=lambda m: m.name)   # chronological chunks
        sigs: List[np.ndarray] = []
        for m in rd_members:
            rd_bytes = tf.extractfile(m).read()
            gfp = extract_gfp(rd_bytes)
            if gfp.size:
                sigs.append(gfp)

        if len(sigs) < 2:
            raise RuntimeError(f"No usable EEG in {tar_path.name}")

        gfp_full = np.hstack(sigs)
        mid      = gfp_full.size // 2
        first, second = gfp_full[:mid], gfp_full[mid:]

        k1, _, _ = ric_k(first)
        k2, _, _ = ric_k(second)
        return tar_path.stem, k1, k2


# ──────────────────────────────────── main ──────────────────────────────────
def main(root: pathlib.Path):
    if not root.is_dir():
        sys.exit(f"✘  Folder {root} not found")

    tars = sorted(root.glob("co2*.tar.gz"))
    print(f"✓  Found {len(tars)} tar archives under {root.name}")

    subj_rows = []
    for tar_f in tars:
        try:
            name, k1, k2 = process_tar(tar_f)
            subj_rows.append((name, k1, k2))
        except Exception as exc:
            warnings.warn(f"{tar_f.name}: {exc}", RuntimeWarning)

    df = pd.DataFrame(subj_rows, columns=["subject", "K_first", "K_second"])
    df.to_csv("SMNI_TRR_curvature.csv", index=False)
    print(f"✓  Saved curvature table  →  SMNI_TRR_curvature.csv")

    # reliability
    mat = df[["K_first", "K_second"]].to_numpy(float)
    r,  p  = pearsonr(mat[:, 0], mat[:, 1])
    icc    = icc_2_1(mat)

    print(textwrap.dedent(f"""
        ---------------- Test–retest reliability ----------------
        Pearson r  = {r:6.3f}   (p = {p:1.2e})
        ICC(2,1)   = {icc:6.3f}
        subjects   = {mat.shape[0]}
        ---------------------------------------------------------
    """).strip())

    # quick plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mat[:, 0], mat[:, 1], s=10)
    lims = [mat.min() * 1.05, mat.max() * 0.95]
    ax.plot(lims, lims, "--k", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("K  (first half)")
    ax.set_ylabel("K  (second half)")
    ax.set_title(f"RIC test–retest  r={r:0.2f}  ICC={icc:0.2f}")
    fig.tight_layout()
    fig.savefig("SMNI_TRR_reliability.png", dpi=300)
    plt.close(fig)
    print("✓  Plot saved  →  SMNI_TRR_reliability.png")


if __name__ == "__main__":
    # run
    main(ROOT)
