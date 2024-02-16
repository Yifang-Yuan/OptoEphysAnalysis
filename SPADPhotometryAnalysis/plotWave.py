# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:53:18 2024

@author: Yifang
"""
import matplotlib.pyplot as plt
import numpy as np

# Frequencies and duty cycles
frequency = 500  # Hz
duty_cycle_1 = 0.2  # 20%
duty_cycle_2 = 0.3  # 30%

# Time vector
t = np.linspace(0, 3 / frequency, 1000, endpoint=False)

# Generate square waves
square_wave_1 = np.where((t * frequency) % 1 < duty_cycle_1, 1, 0)
square_wave_2 = np.where((t * frequency + 0.5) % 1 < duty_cycle_2, 1, 0)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Plot the square waves on separate axes
ax1.plot(t, square_wave_1, label=f'20% Duty Cycle')
ax2.plot(t, square_wave_2, label=f'30% Duty Cycle',color='purple')

# Set labels and title
ax2.set_xlabel('Time (seconds)')
fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical')
fig.suptitle(f'Square Waves - {frequency} Hz, 3 Cycles')

# Set y-axis limits
ax1.set_ylim([-0.2, 1.2])
ax2.set_ylim([-0.2, 1.2])

# Add legends
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
# Remove the frame around each subplot
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
# Show the plot
plt.show()