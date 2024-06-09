# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:48:35 2024

@author: Yang
"""

import matplotlib.pyplot as plt

# Data
x = [30, 70, 110, 150, 190, 230, 270, 310, 350]
y = [0.33, 0.52, 0.75, 1.04, 1.30, 1.52, 1.75, 1.95, 2.2]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, marker='o', linestyle='-', color='b')

# Set the labels
ax.set_xlabel('LED current (mA)', fontsize=14)
ax.set_ylabel('Readout signal (V)', fontsize=14)

# Optionally, you can set the title
ax.set_title('Linearity Test', fontsize=16)

# Remove right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Show the plot
fig.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt

# New data
x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
y = [0.016, 0.008, 0.0055, 0.004, 0.0035, 0.0031, 0.0028]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, marker='o', linestyle='-', color='b')

# Set the labels
ax.set_xlabel('Average Signal (V)', fontsize=14)
ax.set_ylabel('Signal coefficient of variation', fontsize=14)

# Optionally, you can set the title
ax.set_title('Variation test', fontsize=16)

# Remove right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Show the plot
fig.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt

# New data
x = [0, 512, 1024, 1536,2048,2560, 3072,3584, 4096]
y = [0, 13,27, 41,55, 69,83, 96,110]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, marker='o', linestyle='-', color='b')

# Set the labels
ax.set_xlabel('Average Signal (V)', fontsize=14)
ax.set_ylabel('Signal coefficient of variation', fontsize=14)

# Optionally, you can set the title
ax.set_title('Linearity Test', fontsize=16)

# Remove right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Show the plot
fig.tight_layout()
plt.show()
#%%
# New data
x = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
y = [0, 0.01, 0.09, 0.08, 0.17, 0.1, 0.11, 0.23, 0.3]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(x, y, color='b')  # Use scatter plot instead of line plot

# Set the labels
ax.set_xlabel('Average Signal (V)', fontsize=12)
ax.set_ylabel('LED current std. dev. (mV)', fontsize=12)

# Optionally, you can set the title
#ax.set_title('Linearity Test', fontsize=16)

# Remove right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Show the plot
fig.tight_layout()
plt.show()
#%%
# New data
x = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
y = [0.2, 0.22, 0.225, 0.255, 0.28, 0.3, 0.325, 0.355]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 2))
ax.scatter(x, y, color='b')  # Use scatter plot instead of line plot

# Set the labels
ax.set_xlabel('Input Voltage (V)', fontsize=12)
ax.set_ylabel('Noise Std (mV)', fontsize=10)

# Optionally, you can set the title
#ax.set_title('Linearity Test', fontsize=16)

# Remove right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Show the plot
fig.tight_layout()
plt.show()