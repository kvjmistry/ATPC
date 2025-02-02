import glob
import pandas as pd
import pickle
import sys

# Example
# python3 merge_outputs.py /ospool/ap40/data/krishan.mistry/job/ATPC/trackreco/ATPC_0nubb/15bar/nodiff/ /home/krishan.mistry/code/ATPC/merged/ATPC_0nubb_15bar_nodiff

file_path = sys.argv[1]

file_out = sys.argv[2]

files = sorted(glob.glob(f"{file_path}/reco/*.h5"))

dfs = []
df_meta = []

for f in files:
    dfs.append(pd.read_hdf(f, "data"))
    df_meta.append(pd.read_hdf(f, "meta"))

dfs = pd.concat(dfs)
df_meta = pd.concat(df_meta)

print(dfs)
print(df_meta)

with pd.HDFStore(f"{file_out}_reco.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', dfs, format='table')
    store.put('meta', df_meta, format='table')

# Finished with these dataframes
del dfs
del df_meta

Tracks = []
connections = []
connection_counts = []

files = sorted(glob.glob(f"{file_path}/pkl/*.pkl"))

for index, f in enumerate(files):
    with open(f, 'rb') as pickle_file:  # Use 'rb' for reading in binary

        if (index == 0):
            Tracks      = pickle.load(pickle_file)
            connections = pickle.load(pickle_file)
            connection_counts = pickle.load(pickle_file)
        else:
            Tracks.update(pickle.load(pickle_file))
            connections.update(pickle.load(pickle_file))
            connection_counts.update(pickle.load(pickle_file))


with open(f"{file_out}_trackreco.pkl", 'wb') as pickle_file:
    pickle.dump(Tracks, pickle_file)
    pickle.dump(connections, pickle_file)
    pickle.dump(connection_counts, pickle_file)
