# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 21:46:42 2025

@author: yifan
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t, bootstrap

sns.set(style="whitegrid")

def load_all_phase_MI(folder):
    mi_list, amp_list, bin_centers = [], [], None
    for fname in os.listdir(folder):
        if "phase_MI_" in fname and fname.endswith(".pkl"):
            with open(os.path.join(folder, fname), 'rb') as f:
                data = pickle.load(f)
                mi_list.append(data['MI'])
                amp_list.append(data['norm_amp'])
                if bin_centers is None:
                    bin_centers = data['bin_centers']
    return np.array(mi_list), np.vstack(amp_list), bin_centers

def compute_ci(data, confidence=0.95):
    mean = np.mean(data, axis=0)
    error = sem(data, axis=0)
    h = error * t.ppf((1 + confidence) / 2, data.shape[0] - 1)
    return mean, mean - h, mean + h

def plot_multi_phase_tuning_polar(all_means, all_lows, all_highs, bin_centers, animal_ids, save_path=None):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    colors = sns.color_palette("husl", len(all_means))
    for mean, low, high, aid, color in zip(all_means, all_lows, all_highs, animal_ids, colors):
        wrapped_bins = np.append(bin_centers, bin_centers[0])
        ax.plot(wrapped_bins, np.append(mean, mean[0]), label=aid, linewidth=2, color=color)
        ax.fill_between(wrapped_bins, np.append(low, low[0]), np.append(high, high[0]), color=color, alpha=0.3)
    ax.set_yticklabels([]), ax.tick_params(labelsize=14), ax.spines['polar'].set_linewidth(2), ax.grid(False)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10), plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, "phase_tuning_multi_polar.png"), dpi=300, transparent=True)
    plt.show()

def plot_MI_boxplot_animals(animal_mis, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, means, cis_low, cis_high = [], [], [], [], []
    for animal, mis in animal_mis.items():
        data.append(mis)
        labels.append(animal)
        res = bootstrap((np.array(mis),), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
        means.append(np.mean(mis))
        cis_low.append(res.confidence_interval.low)
        cis_high.append(res.confidence_interval.high)
    sns.boxplot(data=data, orient='v', palette="Set2", width=0.6, fliersize=0)
    for i, (mean, low, high) in enumerate(zip(means, cis_low, cis_high)):
        ax.scatter(i, mean, color='black', zorder=5)
        ax.vlines(i, low, high, color='black', linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Modulation Index (MI)", fontsize=14)
    ax.set_title("Modulation Index by Animal", fontsize=16)
    sns.despine(), plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, "MI_boxplot_animals.png"), dpi=300, transparent=True)
    plt.show()

# Main execution
#root_path = 'F:/2025_ATLAS_SPAD/Figure2_Pyr_theta/Theta_Phase_MI_MultipleAnimals'
root_path = 'F:/2025_ATLAS_SPAD/Figure3_Pyr_gamma/GammaPhase_MI_MultipleAnimals'

animal_folders = [os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]

all_mean_amps, all_ci_lows, all_ci_highs, animal_ids, animal_mis = [], [], [], [], {}

for animal_folder in animal_folders:
    animal_id = os.path.basename(animal_folder)
    mi_values, norm_amp_all, bin_centers = load_all_phase_MI(animal_folder)
    mean_amp, ci_low, ci_high = compute_ci(norm_amp_all)
    all_mean_amps.append(mean_amp)
    all_ci_lows.append(ci_low)
    all_ci_highs.append(ci_high)
    animal_ids.append(animal_id)
    animal_mis[animal_id] = mi_values

plot_multi_phase_tuning_polar(all_mean_amps, all_ci_lows, all_ci_highs, bin_centers, animal_ids, save_path=root_path)
plot_MI_boxplot_animals(animal_mis, save_path=root_path)