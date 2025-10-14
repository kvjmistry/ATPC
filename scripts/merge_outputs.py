import glob
import pandas as pd
import pickle
import sys
sys.path.append("../scripts")
sys.path.append("../notebooks")
from TrackReconstruction_functions import *
from reconstruction_functions import *

# Example

# python3 merge_outputs.py ATPC_Bi_ion 1 5percent


mode=sys.argv[1]
pressure=sys.argv[2]
diffusion=sys.argv[3]

print("mode:", mode)
print("pressure:", pressure, "bar")
print("diffusion:", diffusion)

computer = "osg"
computer = "argon"

if computer == "osg":
    file_path = f"/ospool/ap40/data/krishan.mistry/job/ATPC/trackreco/{mode}/{pressure}bar/{diffusion}/"
    file_out = f"/home/krishan.mistry/code/ATPC/merged/{mode}_{pressure}bar_{diffusion}"
else:
    file_path = f"/media/argon/HardDrive_8TB/Krishan/ATPC/trackreco/{mode}/{pressure}bar/{diffusion}/"
    file_out = f"/media/argon/HardDrive_8TB/Krishan/ATPC/trackreco/merged/{mode}_{pressure}bar_{diffusion}"

print("file_path", file_path)
print("file_out", file_out)

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

# ------------------------------------------------------------------------------------------
df_primary = df_meta[ (df_meta.label == "Primary") & (df_meta.primary == 1)]
df_meta, df_primary, cuts = ApplyCuts(df_meta, df_primary, pressure, diffusion, "enr", 1.0)
df_primary = df_primary[ cuts ]
df_meta = df_meta[(df_meta.event_id.isin(df_primary.event_id.unique()))]

filtered_events = df_meta[(df_meta.event_id.isin(df_primary.event_id.unique()))].event_id.unique()

# Only filter 100 events for signal
#if (mode == "ATPC_0nubb"):
# filtered_events = filtered_events[0:100] # for now filter first 100 events for all

if (len(filtered_events) ==0):
    filtered_events = df_meta.event_id.unique()[0:100]

print("Total events to filter:", len(filtered_events))

# ------------------------------------------------------------------------------------------
counter = 0
for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    df_hits = pd.read_hdf(f, "data")
    df_hits = df_hits[df_hits.event_id.isin(filtered_events)]
    if (len(df_hits) > 0):
        counter+=len(df_hits.event_id.unique())
        dfs.append(df_hits)

    if (counter >= len(filtered_events)):
        break

dfs = pd.concat(dfs)

print(dfs)
print("Tot saved events:", len(dfs.event_id.unique()))

with pd.HDFStore(f"{file_out}_reco.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', dfs, format='table')
    store.put('meta', df_meta, format='table')

# Finished with these dataframes
del dfs
del df_meta

# ------------------------------------------------------------------------------------------

Tracks = []
connections = []
connection_counts = []

files = sorted(glob.glob(f"{file_path}/pkl/*.pkl"))

counter = 0
for index, f in enumerate(files):

    if index %50 ==0:
        print(f"{index} /", len(files))

    with open(f, 'rb') as pickle_file:  # Use 'rb' for reading in binary

        # Load data from file
        track_data = pickle.load(pickle_file)
        conn_data = pickle.load(pickle_file)
        conn_count_data = pickle.load(pickle_file)

        # Filter during loading
        track_data = {k: v for k, v in track_data.items() if k in filtered_events}
        conn_data = {k: v for k, v in conn_data.items() if k in filtered_events}
        conn_count_data = {k: v for k, v in conn_count_data.items() if k in filtered_events}
        counter+=len(track_data)

        # Initialize or update dictionaries
        if index == 0:
            Tracks = track_data
            connections = conn_data
            connection_counts = conn_count_data
        else:
            Tracks.update(track_data)
            connections.update(conn_data)
            connection_counts.update(conn_count_data)

        if (counter >= len(filtered_events)):
            break

print("Tot saved events:", counter)

with open(f"{file_out}_trackreco.pkl", 'wb') as pickle_file:
    pickle.dump(Tracks, pickle_file)
    pickle.dump(connections, pickle_file)
    pickle.dump(connection_counts, pickle_file)
