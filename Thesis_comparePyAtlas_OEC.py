import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#%%
# ─── Data ──────────────────────────────────────────────────────────
py_data    = np.array([0.148, 0.111, 0.108, 0.108, 0.098, 0.147, 0.165, 0.143, 0.140])
atlas_data = np.array([0.226, 0.250, 0.229, 0.460, 0.456, 0.206, 0.167, 0.176, 0.283])
mask         = ~np.isnan(py_data) & ~np.isnan(atlas_data)
py_paired    = py_data[mask]
atlas_paired = atlas_data[mask]

def mean_sem(arr):
    return np.nanmean(arr), stats.sem(arr, nan_policy='omit')

py_mean, py_sem       = mean_sem(py_data)
atlas_mean, atlas_sem = mean_sem(atlas_data)

t_stat, p_val = stats.ttest_rel(py_paired, atlas_paired)

# ─── Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 4))                          # compact canvas

rng = np.random.default_rng(42)

ax.scatter(rng.normal(1, 0.04, py_data.size),
           py_data,
           s=25,            # smaller marker (points²)
           color='tab:orange', alpha=0.8, label='pyPhotometry')

ax.scatter(rng.normal(2, 0.04, atlas_data.size),
           atlas_data,
           s=25,            # same smaller marker
           color='tab:blue',  alpha=0.8, label='ATLAS')

# mean ± SEM
ax.errorbar(1, py_mean,    yerr=py_sem,    fmt='none',
            color='k', capsize=6, linewidth=2)
ax.errorbar(2, atlas_mean, yerr=atlas_sem, fmt='none',
            color='k', capsize=6, linewidth=2)

# horizontal lines to mark the means
ax.hlines(py_mean,    0.9, 1.1, color='k', linewidth=3)
ax.hlines(atlas_mean, 1.9, 2.1, color='k', linewidth=3)

# formatting
ax.set_xticks([1, 2])
ax.set_xticklabels(['pyPhotometry(time-div)', 'ATLAS(cont)'], fontsize=14)
ax.set_xlim(0.8, 2.2)                                           # trims blank space
ax.set_ylabel('Peak theta correlation with LFP', fontsize=15)
ax.tick_params(axis='y', labelsize=13)
ax.set_title(f't-test (n=9, p = {p_val:.3f})', fontsize=14, pad=10)

ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(pad=0.4)
#%%
py_data    = np.array([0.043, 0.060, 0.086, 0.039, 0.055, 0.048, 0.034, 0.034])
atlas_data = np.array([0.099, 0.077, 0.114, 0.115, 0.137, 0.100,    np.nan,    np.nan])

mask         = ~np.isnan(py_data) & ~np.isnan(atlas_data)
py_paired    = py_data[mask]
atlas_paired = atlas_data[mask]

def mean_sem(arr):
    return np.nanmean(arr), stats.sem(arr, nan_policy='omit')

py_mean, py_sem       = mean_sem(py_data)
atlas_mean, atlas_sem = mean_sem(atlas_data)

t_stat, p_val = stats.ttest_rel(py_paired, atlas_paired)

# ─── Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 4))                          # compact canvas

rng = np.random.default_rng(42)

ax.scatter(rng.normal(1, 0.04, py_data.size),
           py_data,
           s=25,            # smaller marker (points²)
           color='tab:orange', alpha=0.8, label='pyPhotometry')

ax.scatter(rng.normal(2, 0.04, atlas_data.size),
           atlas_data,
           s=25,            # same smaller marker
           color='tab:blue',  alpha=0.8, label='ATLAS')

# mean ± SEM
ax.errorbar(1, py_mean,    yerr=py_sem,    fmt='none',
            color='k', capsize=6, linewidth=2)
ax.errorbar(2, atlas_mean, yerr=atlas_sem, fmt='none',
            color='k', capsize=6, linewidth=2)

# horizontal lines to mark the means
ax.hlines(py_mean,    0.9, 1.1, color='k', linewidth=3)
ax.hlines(atlas_mean, 1.9, 2.1, color='k', linewidth=3)

# formatting
ax.set_xticks([1, 2])
ax.set_xticklabels(['pyPhotometry', 'ATLAS'], fontsize=14)
ax.set_xlim(0.8, 2.2)                                           # trims blank space
ax.set_ylabel('Peak theta correlation with LFP', fontsize=15)
ax.tick_params(axis='y', labelsize=13)
ax.set_title(f't-test (p = {p_val:.3f})', fontsize=14, pad=10)

ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(pad=0.4)
#%%
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats   # only needed for the z-scoring example
from SPADPhotometryAnalysis import photometry_functions as fp
# ── 1. Locate & load  ──────────────────────────────────────────────
DATA_DIR = pathlib.Path(r'G:\2024_OEC_Atlas_main\1732333_pyramidal_G8f_Atlas\Day1\SyncRecording3')      # <-- adjust
in_file  = DATA_DIR / 'Green_traceAll.csv'
out_file = DATA_DIR / 'Zscore_traceAll.csv'
# If your CSV has a header row, keep skip_header=1; otherwise set it to 0
raw = np.genfromtxt(in_file, delimiter=",", skip_header=1)

# ── 2. Plot traces ────────────────────────────────────────────────
plt.figure(figsize=(9, 3))
plt.plot(raw)                      # each column becomes one trace
plt.xlabel("Sample")
plt.ylabel("Signal (a.u.)")
plt.title("Green_traceAll – raw traces")
plt.tight_layout()
plt.show()

# ── 3. Your calculations here ─────────────────────────────────────
def calc_fn(arr: np.ndarray) -> np.ndarray:
    """Example: column-wise z-score."""
    lambd = 5e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 50
    sig_base=fp.airPLS(raw,lambda_=lambd,porder=porder,itermax=itermax) 
    sig = (raw - sig_base)  
    dff_sig=100*sig / sig_base
    plt.figure(figsize=(9, 3))
    plt.plot(dff_sig)                      # each column becomes one trace
    plt.xlabel("Sample")
    plt.ylabel("Signal (a.u.)")
    plt.title("Green_traceAll – raw traces")
    plt.tight_layout()
    plt.show()

    return dff_sig

z_array = calc_fn(raw)

# ── 4. Save result ────────────────────────────────────────────────
np.savetxt(out_file, z_array, delimiter=",", fmt="%.6f")
print("Saved:", out_file.resolve())