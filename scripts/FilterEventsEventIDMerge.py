import glob
import pandas as pd
import sys
from pathlib import Path
import pickle

# Example
# python3 FilterEventsEventIDMerge.py ATPC_Bi_ion 1bar 5percent
# python3 FilterEventsEventIDMerge.py ATPC_Tl_ion 1bar 5percent
# python3 FilterEventsEventIDMerge.py ATPC_single 1bar 5percent
# python3 FilterEventsEventIDMerge.py ATPC_0nubb 1bar 5percent

mode=sys.argv[1]
pressure=sys.argv[2]
diffusion=sys.argv[3]

# Get input and output folders
base_path = Path(f"/media/argon/HardDrive_8TB/Krishan/ATPC//trackreco/{mode}/{pressure}/{diffusion}/")
input_dir = base_path / f"reco"
output_dir = base_path / f"reco_filtered"

output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.h5"))

# Load in the event list
event_list = pd.read_csv(f"../eventlists/ATPC_{pressure}_{diffusion}.csv");
event_list = event_list[event_list.subType == mode].event_id.values # select events 

# ------------------------------------------------------------------------------------------

df_hits_all = []

counter = 0
for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    f = Path(f)

    df_hits  = pd.read_hdf(f, "data")
    df_hits  = df_hits[df_hits.event_id.isin(event_list)]
    counter+=len(df_hits.event_id.unique())

    df_hits_all.append(df_hits)
    print("Tot saved events:", len(df_hits.event_id.unique()), counter)
    if counter > 35000:
        break

df_hits_all = pd.concat(df_hits_all)

# new filename: ATPC_0nubb_0_filtered.h5
outfile = output_dir / f"{mode}_{pressure}_{diffusion}_filtered.h5"
print(f"{f} → {outfile}")

with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/hits', df_hits_all, format='table')
    
    
# ------------------------------------------------------------------------------------------

Tracks = []
connections = []
connection_counts = []

files = sorted(glob.glob(f"/media/argon/HardDrive_8TB/Krishan/ATPC//trackreco/{mode}/{pressure}/{diffusion}/pkl/*.pkl"))

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
        track_data = {k: v for k, v in track_data.items() if k in event_list}
        conn_data = {k: v for k, v in conn_data.items() if k in event_list}
        conn_count_data = {k: v for k, v in conn_count_data.items() if k in event_list}
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

        if (counter >= len(event_list)):
            break

print("Tot saved events:", counter)

file_out = output_dir / f"{mode}_{pressure}_{diffusion}_pkl_filtered.h5"
print(f"{file_out}")
with open(f"{file_out}", 'wb') as pickle_file:
    pickle.dump(Tracks, pickle_file)
    pickle.dump(connections, pickle_file)
    pickle.dump(connection_counts, pickle_file)
