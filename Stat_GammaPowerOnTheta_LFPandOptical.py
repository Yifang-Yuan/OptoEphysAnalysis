"""
Paired comparison within each state:
  LFP γ preferred θ phase  vs  Optical γ preferred θ phase
- One test & figure per group (Locomotion, Awake-stationary, REM)
- Paired circular permutation test (sign-flip of paired angular diffs)
- Prism-style paired boxplots with connecting lines and p-value bar.

Author: yifan (assembled by ChatGPT) — Δμ reporting wrapped to (-180°, 180°]
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip

# =================== USER PATHS ===================
LOCOMOTION_DIR = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\OpenField_DLCtracking\GammaPowerOnTheta")
AWAKE_DIR      = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\GammaPowerOnTheta")
REM_DIR        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleep\GammaPowerOnTheta")

PATTERN        = "GammaPowerOnTheta_trial*.pkl.gz"
OUT_DIR        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\GammaPhase_Paired_LFP_vs_Opt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM         = 5000     # permutations for paired circular test
ALPHA          = 0.05
# ==================================================


# ----------------- IO helpers -----------------
def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_gamma_group(folder: Path, pattern=PATTERN):
    """
    Read all pickles in a folder and extract paired per-trial:
      - lfp_phi (rad), opt_phi (rad)
    Returns only trials present in the folder (paired by file).
    """
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise ValueError(f"No files matching {pattern} in {folder}")

    lfp_phi, opt_phi = [], []
    for p in paths:
        rec = _load_pickle(p)
        lfp_phi.append(float(rec["lfp"]["preferred_phase_rad"]))
        opt_phi.append(float(rec["opt"]["preferred_phase_rad"]))
    lfp_phi = (np.array(lfp_phi) % (2*np.pi)).astype(float)
    opt_phi = (np.array(opt_phi) % (2*np.pi)).astype(float)
    return {"lfp_phi": lfp_phi, "opt_phi": opt_phi, "n": len(paths), "paths": paths}


# -------------- Circular helpers --------------
def circ_mean(alpha):
    """Mean direction (rad) + resultant length of the mean (Rbar)."""
    C = np.nanmean(np.cos(alpha))
    S = np.nanmean(np.sin(alpha))
    mu = (np.arctan2(S, C)) % (2*np.pi)
    Rbar = np.sqrt(C**2 + S**2)
    return mu, Rbar

def circ_dist(a, b):
    """Smallest signed circular difference a-b in (-pi, pi]."""
    return (a - b + np.pi) % (2*np.pi) - np.pi

def wrap_angles_for_plot(phi_rad, ref_rad=0.0, deg_range=(-180.0, 180.0)):
    """Rotate by ref_rad and wrap to [low, high] degrees for plotting only."""
    low, high = deg_range
    width = high - low
    shifted = (phi_rad - ref_rad) % (2*np.pi)
    deg = np.degrees(shifted)
    return ((deg - low) % width) + low

def wrap_pm180(deg):
    """Wrap degrees to (-180, 180]."""
    d = ((deg + 180.0) % 360.0) - 180.0
    return 180.0 if np.isclose(d, -180.0) else d

# -------------- Paired circular permutation test --------------
def paired_circ_perm_test(phi1, phi2, n_perm=5000, rng=None):
    """
    Paired circular permutation (sign-flip) test.
    Tests H0: mean signed circular difference == 0.
    Test statistic = |circular mean of paired diffs| (radians).

    Returns:
        p_value,
        abs_diff_deg  -> |Δμ| in degrees  (using wrapped signed mean),
        signed_deg    -> Δμ in degrees, wrapped to (-180, 180]
    """
    if rng is None: rng = np.random.default_rng()
    phi1 = np.asarray(phi1) % (2*np.pi)
    phi2 = np.asarray(phi2) % (2*np.pi)
    if len(phi1) != len(phi2):
        raise ValueError("Paired arrays must have equal length.")

    d = circ_dist(phi1, phi2)            # signed diffs in (-pi, pi]
    mu_obs, _ = circ_mean(d)             # circ_mean returns [0, 2π)
    # Convert to signed in (-π, π]
    mu_obs_signed = (mu_obs + np.pi) % (2*np.pi) - np.pi
    signed_deg = wrap_pm180(np.degrees(mu_obs_signed))
    abs_diff_deg = abs(signed_deg)

    # sign-flip permutations (flip within each pair => d -> ±d)
    obs = abs(mu_obs_signed)
    count_ge = 0
    for _ in range(n_perm):
        flips = rng.integers(0, 2, size=len(d)) * 2 - 1   # ±1
        d_perm = d * flips
        mu_p, _ = circ_mean(d_perm)
        mu_p_signed = (mu_p + np.pi) % (2*np.pi) - np.pi
        stat = abs(mu_p_signed)
        if stat >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, float(abs_diff_deg), float(signed_deg)

def signed_delta_mu_deg(phi_lfp, phi_opt):
    mu_lfp, _ = circ_mean(phi_lfp)
    mu_opt, _ = circ_mean(phi_opt)
    d_rad = circ_dist(mu_opt, mu_lfp)      # in (-pi, pi]
    return wrap_pm180(np.degrees(d_rad))   # in (-180, 180]


# -------------- Plot helpers --------------
def jitter(n, width=0.07, rng=None):
    if rng is None: rng = np.random.default_rng()
    return rng.uniform(-width, width, size=n)

def style_ax(ax, fontsize=16):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize + 2)

def smart_ylim(data_list, pad_frac=0.06):
    all_data = np.concatenate([np.asarray(d) for d in data_list])
    dmin, dmax = np.nanmin(all_data), np.nanmax(all_data)
    if np.isclose(dmin, dmax):
        return dmin - 1, dmax + 1
    pad = (dmax - dmin) * pad_frac
    return dmin - pad, dmax + pad

def p_to_asterisks(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

def add_pair_bar(ax, x1, x2, data_list, p, text_fs=16, line_h_frac=0.06, pad_top_frac=0.04):
    """Single Prism-style bar between x1 & x2; auto-extends ylim so label isn’t cramped."""
    y0, y1 = ax.get_ylim()
    data_max = max(np.nanmax(d) for d in data_list)
    span = (y1 - y0) if (y1 > y0) else 1.0
    h = span * line_h_frac * 0.5
    base = max(y1, data_max) + span * pad_top_frac

    ax.plot([x1, x1, x2, x2], [base, base + h, base + h, base], lw=1.8, c='k', clip_on=False)
    ax.text((x1 + x2) / 2, base + h, p_to_asterisks(p), ha='center', va='bottom',
            fontsize=text_fs, clip_on=False)
    ax.set_ylim(y0, base + h + span * 0.02)

def paired_boxplot(phi1_rad, phi2_rad, title, outfile,
                   label1="LFP γ pref θ phase", label2="Optical γ pref θ phase",
                   fontsize=16):
    # now returns wrapped Δμ
    p, abs_diff_deg, signed_deg = paired_circ_perm_test(phi1_rad, phi2_rad, n_perm=N_PERM)

    grand_mu, _ = circ_mean(np.concatenate([phi1_rad, phi2_rad]))
    deg1 = wrap_angles_for_plot(phi1_rad, ref_rad=grand_mu)
    deg2 = wrap_angles_for_plot(phi2_rad, ref_rad=grand_mu)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    positions = [1, 2]
    data_deg = [deg1, deg2]
    ax.boxplot(data_deg, positions=positions, widths=0.5, showfliers=False)

    # scatter + lines
    x1 = np.full_like(deg1, positions[0], dtype=float) + jitter(len(deg1), 0.06)
    x2 = np.full_like(deg2, positions[1], dtype=float) + jitter(len(deg2), 0.06)

    # LFP points: dark grey
    ax.scatter(x1, deg1, s=55, alpha=0.9, zorder=3, color="0.2")
    # Optical points: green
    ax.scatter(x2, deg2, s=55, alpha=0.9, zorder=3, color="green")

    # Connecting lines
    for xi, yi, xj, yj in zip(x1, deg1, x2, deg2):
        ax.plot([xi, xj], [yi, yj], color='k', alpha=0.25, lw=1.2, zorder=2)

    ax.set_xticks(positions)
    ax.set_xticklabels([label1, label2], rotation=10)
    ax.set_ylabel("Preferred θ phase (deg, wrapped)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize+2)

    ax.set_ylim(*smart_ylim(data_deg, pad_frac=0.08))
    style_ax(ax, fontsize=fontsize)

    # --- Lower bar so it sits closer to the boxes ---
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    bar_height = max(np.max(d) for d in data_deg) + span * 0.05
    h = span * 0.03
    ax.plot([positions[0], positions[0], positions[1], positions[1]],
            [bar_height, bar_height + h, bar_height + h, bar_height],
            lw=1.7, c='k', clip_on=False)

    # --- Title line with wrapped Δμ above bar ---
    ax.text(np.mean(positions), bar_height + h + span * 0.01,
            f"p = {p:.3g}   (Δμ = {signed_deg:+.1f}°, |Δμ| = {abs_diff_deg:.1f}°)",
            ha='center', va='bottom', fontsize=fontsize)

    # Adjust ylim to fit text comfortably
    ax.set_ylim(y0, bar_height + h + span * 0.12)

    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    print(f"✅ Saved: {outfile}")
    return p, abs_diff_deg, signed_deg


# ----------------- Load groups -----------------
G_loc = load_gamma_group(LOCOMOTION_DIR)
G_aw  = load_gamma_group(AWAKE_DIR)
G_rem = load_gamma_group(REM_DIR)

# ----------------- Run & plot (3 separate figures) -----------------
print("\n=== Paired LFP vs Optical γ preferred θ phase (within state) ===")

pL, dL_abs, dL_signed = paired_boxplot(
    G_loc["lfp_phi"], G_loc["opt_phi"],
    title="Locomotion",
    outfile=OUT_DIR / "Paired_Phase_Locomotion.png"
)

pA, dA_abs, dA_signed = paired_boxplot(
    G_aw["lfp_phi"], G_aw["opt_phi"],
    title="Awake-stationary",
    outfile=OUT_DIR / "Paired_Phase_Awake.png"
)

pR, dR_abs, dR_signed = paired_boxplot(
    G_rem["lfp_phi"], G_rem["opt_phi"],
    title="REM sleep",
    outfile=OUT_DIR / "Paired_Phase_REM.png"
)

# ----------------- Console summary -----------------
def line(name, p, d_signed, d_abs):
    print(f"{name:>16s} : p = {p:.4g} | Δμ = {d_signed:+.1f}° | |Δμ| = {d_abs:.1f}°")

print("Results (paired circular permutation, LFP vs Optical):")
line("Locomotion",        pL, dL_signed, dL_abs)
line("Awake-stationary",  pA, dA_signed, dA_abs)
line("REM",               pR, dR_signed, dR_abs)