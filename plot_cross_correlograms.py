import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import pynapple as nap
import seaborn as sns


def make_plot_for_cells_on_tetrode(spatial_firing, tetrode, figure_folder_path, sampling_rate, windowsize, binsize, color):
    neurons_on_tetrode = spatial_firing[spatial_firing.tetrode == tetrode]
    number_of_neurons = len(neurons_on_tetrode)
    if number_of_neurons > 1:
        cluster_ids = neurons_on_tetrode.cluster_id.values
        fig, axs = plt.subplots(number_of_neurons, number_of_neurons)
        for cluster1_index, cluster1 in enumerate(cluster_ids):
            for cluster2_index, cluster2 in enumerate(cluster_ids):
                neuron_1_times = np.array(
                    (spatial_firing[spatial_firing.cluster_id == cluster1].firing_times.values / sampling_rate)[0])
                neuron_2_times = np.array(
                    (spatial_firing[spatial_firing.cluster_id == cluster2].firing_times.values / sampling_rate)[0])
                cross_corr, xt = nap.cross_correlogram(neuron_1_times, neuron_2_times, binsize=binsize,
                                                       windowsize=windowsize)
                axs[cluster1_index, cluster2_index].bar(xt, cross_corr, binsize, color=color)
                # axs[cluster1_index, cluster2_index].axvline(0, color='black', linewidth=2, linestyle='--')
                sns.despine(top=True, right=True, left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(figure_folder_path + spatial_firing.session_id.iloc[0] + '_tetrode_' + str(
        tetrode) + '_cross_correlograms' + str(windowsize) + '.png')
    plt.close()


def plot_cross_correlograms(recording_path, sampling_rate=30000, binsize=0.02, windowsize=0.5, color='steelblue'):
    """
    This script will make a new folder in MountainSort/Figures/cross_correlograms and add a plot for each tetrode.
    All combinations of cross-correlograms will be plotted between cells of a tetrode including autocorrelograms.
    sampling_rate: sampling rate of electrphysiology data (Hz)
    binsize: bin size for making correlogram plots (sec)
    windowsize: this is half the correlogram plot (sec)
    """

    path_to_spatial_firing = recording_path + "/MountainSort/DataFrames/spatial_firing.pkl"
    if os.path.exists(path_to_spatial_firing):
        figure_folder_path = recording_path + "/MountainSort/Figures/cross_correlograms/"
        if not os.path.exists(figure_folder_path):
            os.mkdir(figure_folder_path)
        # plot all combinations
        spatial_firing = pd.read_pickle(path_to_spatial_firing)
        tetrode_ids = spatial_firing.tetrode.unique()
        for tetrode in tetrode_ids:
            make_plot_for_cells_on_tetrode(spatial_firing, tetrode, figure_folder_path, sampling_rate, windowsize,
                                           binsize, color)


def process_recordings(experiment_folder, sampling_rate=30000, binsize=0.02, windowsize=0.5, color='midnightblue'):
    print('I will plot cross-correlograms for all the recordings within tetrodes from this folder: ' + experiment_folder)
    print('The results will be in MountainSort/Figures/cross_correlograms/ for each recording')
    recording_list = [f.path for f in os.scandir(experiment_folder) if f.is_dir()]
    for recording in recording_list:
        plot_cross_correlograms(recording, sampling_rate=sampling_rate, binsize=binsize, windowsize=windowsize, color=color)


def main():
    # all the recordings in this folder will be processed
    experiment_folder = "/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/"
    # adjust parameters here (not the sampling rate)
    # process_recordings(experiment_folder, sampling_rate=30000, binsize=0.02, windowsize=0.5, color='midnightblue')
    process_recordings(experiment_folder, sampling_rate=30000, binsize=0.002, windowsize=0.1, color='midnightblue')
    # if you just want to make it for a single recording:
    # recording_path = "/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/M3_2021-06-16_14-10-45_of/"
    # identify_interneurons(recording_path)


if __name__ == '__main__':
    main()
