# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:02:57 2023

@author: Yifang

"""
import pandas as pd

dpath="G:/SPAD/SPADData/20230818_SPADOEC/SyncRecording0_1681633_g8m/"
filename='ZippedTracking_0.csv'
savename='AnimalTracking_0.csv'

dataA = pd.read_csv(dpath+filename)
dataB = dataA.reindex(dataA.index.repeat(2)).reset_index(drop=True)
dataB.to_csv(dpath+savename, index=False)


#%%
import pandas as pd
import numpy as np

# Create a sample DataFrame with X and Y coordinates
data = {'Time': [1, 2, 3, 4, 5],
        'X': [0, 1, 3, 6, 9],
        'Y': [0, 2, 4, 7, 10]}

df = pd.DataFrame(data)

# Calculate the Euclidean distance (speed) between consecutive points
dataf = np.sqrt(np.diff(df['X'])**2 + np.diff(df['Y'])**2)

# Add NaN in the first row to match the original DataFrame length
df['Speed'] = [np.nan] + dataf.tolist()

# Handle missing values (NaN) by filling them with a specific value (e.g., 0)
df['Speed'].fillna(0, inplace=True)

# Print the DataFrame
print(df)