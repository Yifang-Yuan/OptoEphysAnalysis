# --- Requirements ---
# pip install numpy pandas matplotlib seaborn scipy
# (and make sure your OpenEphysTools module is on PYTHONPATH)
# -----------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, detrend
import OpenEphysTools as OE  # <-- your toolkit

# ========= USER INPUTS =========
PKL_PATH     = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\SyncRecording5/Ephys_tracking_photometry_aligned.pkl"  # <- change if needed
LFP_CHANNEL  = "LFP_1"
START_TIME   = 3   # seconds (relative to df['timestamps'])
END_TIME     = 23  # seconds
SPAD_CUTOFF  = 50.0     # Hz, photometry smoothing LPF before HPF(2.5 Hz)
LFP_CUTOFF   = 100.0    # Hz, LFP LPF before HPF(4 Hz)
SPECTRO_YLIM = 20.0     # Hz, display limit for both wavelet plots
SAVE_FIG     = True
OUTFILE      = "example_trace_powerspectral_OE.png"
# =================================

# ---------- helpers ----------
def infer_fs(t_seconds: np.ndarray) -> float:
    dt = np.median(np.diff(t_seconds))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Could not infer sampling rate from timestamps.")
    return 1.0 / dt

def butter_filter(x, btype, cutoff, fs, order=3):
    nyq = fs / 2.0
    wn = np.asarray(cutoff) / nyq if np.iterable(cutoff) else cutoff / nyq
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x)

def zscore(x):
    x = np.asarray(x)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (x - mu) / sd


# ---------- load & slice (relative 0-based time; index-safe) ----------
df = pd.read_pickle(PKL_PATH)
if "timestamps" not in df.columns:
    raise KeyError("DataFrame must contain a 'timestamps' column in seconds.")

# Infer Fs and make a 0-based relative time just for sanity/printing
t_abs  = df["timestamps"].to_numpy(dtype=float)
Fs     = infer_fs(t_abs)
t_rel  = t_abs - t_abs[0]                 # starts at 0


# Convert requested [START_TIME, END_TIME] to sample indices
if END_TIME <= START_TIME:
    raise RuntimeError("END_TIME must be greater than START_TIME.")

n_total = len(t_rel)
idx_start = int(np.searchsorted(t_rel, START_TIME, side="left"))
idx_end_req = int(np.searchsorted(t_rel, END_TIME,   side="right"))

# Clamp to available samples
idx_start = max(0, min(idx_start, n_total - 1))
idx_end   = max(idx_start + 1, min(idx_end_req, n_total))

# Build a perfectly regular time axis from sample indices
idx = np.arange(idx_start, idx_end)
ts  = (idx - idx_start) / Fs                 # starts at 0, exact length via indices

# Optional: warn if truncated by file length
req_dur   = END_TIME - START_TIME
actual_dur = (idx_end - idx_start) / Fs
if actual_dur + 1e-9 < req_dur:
    print(f"Note: requested {req_dur:.3f}s, truncated to {actual_dur:.3f}s "
          f"(file length is {t_rel[-1]:.3f}s).")

# Channels (use iloc with absolute indices for perfect alignment)
lfp = df.iloc[idx_start:idx_end][LFP_CHANNEL].to_numpy(dtype=float)
if "zscore_raw" in df.columns:
    spad = df.iloc[idx_start:idx_end]["zscore_raw"].to_numpy(dtype=float)
else:
    spad = zscore(detrend(df.iloc[idx_start:idx_end]["sig_raw"].to_numpy(dtype=float)))

speed = (df.iloc[idx_start:idx_end]["speed"].to_numpy(dtype=float)
         if "speed" in df.columns else None)

# ---------- filtering (match your class flow) ----------
# Photometry: LPF (SPAD_CUTOFF) -> HPF 2.5 Hz
spad_lpf  = butter_filter(spad, btype="low",  cutoff=SPAD_CUTOFF, fs=Fs, order=3)
spad_filt = butter_filter(spad_lpf, btype="high", cutoff=2.5,        fs=Fs, order=3)

# LFP: LPF (LFP_CUTOFF) -> HPF 4 Hz; convert to mV before plotting/wavelet
lfp_lpf   = butter_filter(lfp,  btype="low",  cutoff=LFP_CUTOFF, fs=Fs, order=5)
lfp_filt  = butter_filter(lfp_lpf, btype="high", cutoff=4.0,     fs=Fs, order=3) / 1000.0

# ---------- colours ----------
spad_color  = sns.color_palette("husl", 8)[3]
lfp_color   = sns.color_palette("husl", 8)[5]
speed_color = "k"

# ---------- plot (OpenEphysTools wavelets) ----------
nrows = 5 if speed is not None else 4
fig, ax = plt.subplots(nrows, 1, figsize=(16, 10))

# 1) SPAD trace (your style function)
# If your OE.plot_trace_in_seconds_ax expects a pandas Series, wrap it:
spad_series = pd.Series(spad_filt, index=pd.Index(ts, name="time_s"))
OE.plot_trace_in_seconds_ax(ax[0], spad_series, Fs, label="photometry",
                            color=spad_color, ylabel="z-score", xlabel=False)

# 2) SPAD wavelet using your OE functions
sst_s, freq_s, pow_s, gws_s = OE.Calculate_wavelet(spad_series, lowpassCutoff=100, Fs=Fs, scale=40)
OE.plot_wavelet(ax[1], sst_s, freq_s, pow_s, Fs=Fs, colorBar=False, logbase=False)

# 3) LFP trace (mV)
lfp_series = pd.Series(lfp_filt, index=pd.Index(ts, name="time_s"))
OE.plot_trace_in_seconds_ax(ax[2], lfp_series, Fs, label="LFP",
                            color=lfp_color, ylabel="mV", xlabel=False)

# 4) LFP wavelet (500 Hz lowpass cutoff inside OE.Calculate_wavelet)
sst_l, freq_l, pow_l, gws_l = OE.Calculate_wavelet(lfp_series, lowpassCutoff=500, Fs=Fs, scale=40)
OE.plot_wavelet(ax[3], sst_l, freq_l, pow_l, Fs=Fs, colorBar=True, logbase=False)

# 5) Speed (black)
if speed is not None:
    ax[4].plot(ts, speed, color=speed_color, linewidth=2.0)
    ax[4].set_ylabel("Speed", fontsize=18)
    ax[4].set_xlabel("Time (s)", fontsize=18)

    # Remove top/right spines
    ax[4].spines['top'].set_visible(False)
    ax[4].spines['right'].set_visible(False)
    # Remove margins
    ax[4].margins(x=0, y=0)
    # Larger tick labels
    ax[4].tick_params(axis='both', labelsize=18, width=1.2)
else:
    ax[3].set_xlabel("Time (s)", fontsize=18)

# Match your spectro y-limits
for a in (ax[1], ax[3]):
    a.set_ylim(0, SPECTRO_YLIM)

# Cosmetics similar to your function
for a in (ax[1], ax[3]):
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)
ax[2].spines['bottom'].set_visible(False)
ax[1].set_xticks([]); ax[1].set_xlabel('')
ax[2].set_xticks([]); ax[2].set_xlabel('')
ax[0].legend().set_visible(False)
ax[2].legend().set_visible(False)
# Ticks & fonts (lightweight)
for a in ax:
    a.tick_params(axis='both', labelsize=16, width=1.2)

fig.tight_layout()

# Save
if SAVE_FIG:
    outdir = os.path.join(os.getcwd(), "makefigure")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, OUTFILE)
    fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    print(f"Saved: {outpath}")

plt.show()
