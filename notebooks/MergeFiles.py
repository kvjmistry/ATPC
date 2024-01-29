# Python script to merge files from the ATPC jobs

# Notebook to slim the production files for the LPR
import os
import sys
import pandas as pd

# Now load in the stack 
directory_path = sys.argv[1]
outputfile = sys.argv[2]

# hits = pd.read_hdf(os.path.join(directory_path, '*.h5'), key = 'hits')

# List comprehension to read DataFrames from HDF5 files and store them in a list
dfs = [pd.read_hdf(os.path.join(directory_path, file), key = 'hits').assign(source_file=file)
       for file in os.listdir(directory_path) if file.endswith('.h5')]

# Concatenate DataFrames into a single DataFrame
hits = pd.concat(dfs, ignore_index=True)

hits = hits.drop(columns=['source_file'])

# Open the HDF5 file in write mode
with pd.HDFStore(outputfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('hits', hits, format='table')

print(hits)