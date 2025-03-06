import glob
import pandas as pd
import pickle
import sys

# Example
# python3 merge_outputs.py /ospool/ap40/data/krishan.mistry/job/ATPC/trackreco/ATPC_0nubb/1bar/nodiff/ /home/krishan.mistry/code/ATPC/merged/ATPC_0nubb_1bar_nodiff

file_path = sys.argv[1]
file_out = sys.argv[2]

files = sorted(glob.glob(f"{file_path}/reco/*.h5"))

dfs = []
df_meta = []

for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    # dfs.append(pd.read_hdf(f, "data"))
    df_meta.append(pd.read_hdf(f, "meta"))

df_meta = pd.concat(df_meta)


print(df_meta)

# Apply some cuts to remove events
df_primary = df_meta[ (df_meta.label == "Primary") & (df_meta.primary == 1)]
cuts = (df_primary.blob2R > 0.4) & \
       (df_primary.blob2 > 0.4) & \
       (df_primary.blob1R > 0.4) & \
       (df_primary.energy > 2.4) & \
       (df_primary.energy < 2.5) & \
       (df_primary.Tortuosity2 > 2) & \
       (df_primary.Squiglicity2 > 1)

df_primary = df_primary[ cuts ]
filtered_events = df_meta[(df_meta.event_id.isin(df_primary.event_id.unique()))].event_id.unique()

for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    df_hits = pd.read_hdf(f, "data")
    df_hits = df_hits[df_hits.event_id.isin(filtered_events)]
    if (len(df_hits) > 0):
        dfs.append(df_hits)

dfs = pd.concat(dfs)

print(dfs)
print("Tot saved events:", dfs.event_id.unique())

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

        # Load data from file
        track_data = pickle.load(pickle_file)
        conn_data = pickle.load(pickle_file)
        conn_count_data = pickle.load(pickle_file)

        # Filter during loading
        track_data = {k: v for k, v in track_data.items() if k in filtered_events}
        conn_data = {k: v for k, v in conn_data.items() if k in filtered_events}
        conn_count_data = {k: v for k, v in conn_count_data.items() if k in filtered_events}

        # Initialize or update dictionaries
        if index == 0:
            Tracks = track_data
            connections = conn_data
            connection_counts = conn_count_data
        else:
            Tracks.update(track_data)
            connections.update(conn_data)
            connection_counts.update(conn_count_data)

with open(f"{file_out}_trackreco.pkl", 'wb') as pickle_file:
    pickle.dump(Tracks, pickle_file)
    pickle.dump(connections, pickle_file)
    pickle.dump(connection_counts, pickle_file)
