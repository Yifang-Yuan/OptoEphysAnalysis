# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 13:48:50 2025

@author: yifan
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t
from scipy.stats import bootstrap

def load_all_phase_MI(folder):
    """Load all phase_MI_*.pkl files in a folder and return lists of MIs and norm_amp arrays."""
    mi_list = []
    amp_list = []

    for fname in os.listdir(folder):
        if "phase_MI_" in fname and fname.endswith(".pkl"):
            with open(os.path.join(folder, fname), 'rb') as f:
                data = pickle.load(f)
                mi_list.append(data['MI'])
                amp_list.append(data['norm_amp'])

    return np.array(mi_list), np.vstack(amp_list), data['bin_centers']

def compute_ci(data, confidence=0.95):
    """Compute mean and confidence interval for a set of vectors."""
    mean = np.mean(data, axis=0)
    n = data.shape[0]
    error = sem(data, axis=0)
    h = error * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

def plot_phase_tuning_polar(bin_centers, mean_amp, lower_ci, upper_ci, save_path=None):
    """Plot polar tuning curve with confidence band, and optionally save as PNG with transparent background."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    wrapped_bins = np.append(bin_centers, bin_centers[0])
    wrapped_mean = np.append(mean_amp, mean_amp[0])
    wrapped_lower = np.append(lower_ci, lower_ci[0])
    wrapped_upper = np.append(upper_ci, upper_ci[0])

    ax.plot(wrapped_bins, wrapped_mean, color="#1b9e77", linewidth=3)
    ax.fill_between(wrapped_bins, wrapped_lower, wrapped_upper, color="#1b9e77", alpha=0.3)

    ax.set_yticklabels([])
    ax.tick_params(labelsize=16)
    ax.spines['polar'].set_linewidth(3)
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(os.path.join(save_path, "phase_tuning_polar.png"), dpi=300, transparent=True)
    plt.show()

def plot_MI_distribution(mi_values, save_path=None):
    """Plot distribution of modulation indices and optionally save as PNG with transparent background."""
    fig = plt.figure(figsize=(6, 4))
    sns.histplot(mi_values, kde=True, bins=15, color="#7570b3")

    plt.xlabel("Modulation Index (MI)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Distribution of Modulation Indices", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(os.path.join(save_path, "MI_distribution.png"), dpi=300, transparent=True)

    plt.show()
    
def plot_MI_boxplot(mi_values, save_path=None):
    """Plot boxplot of modulation indices with mean and confidence interval."""
    fig, ax = plt.subplots(figsize=(4, 4))

    # Calculate mean and 95% CI using bootstrapping
    mi_values = np.array(mi_values)
    res = bootstrap((mi_values,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
    mean_val = np.mean(mi_values)
    ci_low, ci_high = res.confidence_interval

    # Boxplot
    sns.boxplot(data=mi_values, orient='v', color="#7570b3", width=0.4, fliersize=0)

    # Mean point
    ax.scatter(0, mean_val, color='black', zorder=5, label='Mean')
    ax.vlines(x=0, ymin=ci_low, ymax=ci_high, color='black', linestyle='-', linewidth=2, label='95% CI')

    # Styling
    ax.set_xticks([])
    ax.set_ylabel("Modulation Index (MI)", fontsize=14)
    ax.set_title("Modulation Index Boxplot", fontsize=16)
    ax.tick_params(axis='y', labelsize=12)
    sns.despine()
    plt.tight_layout()

    # Save figure
    if save_path:
        fig.savefig(os.path.join(save_path, "MI_boxplot.png"), dpi=300, transparent=True)

    plt.show()

# --- Main Usage ---
folder_path ='F:/2025_ATLAS_SPAD/Figure3_Pyr_gamma/GammaPhase_MI_MultipleAnimals/1844609'  # change this
mi_values, norm_amp_all, bin_centers = load_all_phase_MI(folder_path)
mean_amp, ci_low, ci_high = compute_ci(norm_amp_all)

plot_phase_tuning_polar(bin_centers, mean_amp, ci_low, ci_high,save_path=folder_path)
plot_MI_distribution(mi_values,save_path=folder_path)
plot_MI_boxplot(mi_values, save_path=folder_path)
#%%

folder_path ='F:/2025_ATLAS_SPAD/Figure2_Pyr_theta/Theta_Phase_MI_MultipleAnimals/1844609'  # change this
mi_values, norm_amp_all, bin_centers = load_all_phase_MI(folder_path)
mean_amp, ci_low, ci_high = compute_ci(norm_amp_all)

plot_phase_tuning_polar(bin_centers, mean_amp, ci_low, ci_high,save_path=folder_path)
plot_MI_distribution(mi_values,save_path=folder_path)
plot_MI_boxplot(mi_values, save_path=folder_path)