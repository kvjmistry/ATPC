import glob
import pandas as pd
import pickle
import sys


file_path = sys.argv[1]

file_out = sys.argv[2]

files = sorted(glob.glob(f"{file_path}/*.h5"))

dfs = []

for f in files:
    dfs.append(pd.read_hdf(f))

dfs = pd.concat(dfs)

print(dfs)

Tracks_ = []
connections_ = []
connection_counts_ = []

files = sorted(glob.glob(f"{file_path}/*.pkl"))

for index, f in enumerate(files):
    with open(f, 'rb') as pickle_file:  # Use 'rb' for reading in binary

        if (index == 0):
            Tracks_      = pickle.load(pickle_file)
            connections_ = pickle.load(pickle_file)
            connection_counts_ = pickle.load(pickle_file)
        else:
            Tracks_.update(pickle.load(pickle_file))
            connections_.update(pickle.load(pickle_file))
            connection_counts_.update(pickle.load(pickle_file))


df.to_hdf(f"{file_out}_reco.h5", key='data', mode='w')


with open(f"{file_out}_trackreco.pkl", 'wb') as pickle_file:
    pickle.dump(Track_dict, pickle_file)
    pickle.dump(connected_nodes_dict, pickle_file)
    pickle.dump(connections_count_dict, pickle_file)
