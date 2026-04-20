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

filt_stage="MLP"
# filt_stage="GNN"

# Get input and output folders
# base_path = Path(f"/media/argon/HardDrive_8TB/Krishan/ATPC//trackreco/{mode}/{pressure}/{diffusion}/")
base_path = Path(f"/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//{mode}/{pressure}/{diffusion}/")

if diffusion == "nexus":
    input_dir = base_path / f""
    output_dir = base_path / f"merged/"
else:
    input_dir = base_path / f"reco"
    output_dir = base_path / f"reco_filtered"

output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("*.h5"))
# files = glob.glob("/media/argon/HardDrive_8TB/Krishan/ATPC/ATPC_Tl_ion/1bar/nexus/*.h5")

# Load in the event list
if (filt_stage =="MLP"):
    # listin=f"../eventlists/ATPC_{pressure}_{diffusion}.csv"
    # listin=f"../eventlists/ATPC_{pressure}_{diffusion}_highstats.csv"
    listin=f"../eventlists/ATPC_{pressure}_5percent_highstats.csv"
else:
    listin=f"../eventlists/ATPC_{pressure}_{diffusion}_GNN.csv"

event_list = pd.read_csv(listin);
event_list = event_list[event_list.subType == mode].event_id.values # select events 

# ------------------------------------------------------------------------------------------

df_hits_all = []
df_parts_all = []

counter = 0
for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    f = Path(f)

    if (diffusion == "nexus"):
        # df_hits  = pd.read_hdf(f, "MC/hits")
        df_parts  = pd.read_hdf(f, "MC/particles")
        df_parts  = df_parts[df_parts.event_id.isin(event_list)]
        df_parts = df_parts[df_parts.primary == 1]
        df_parts_all.append(df_parts)
    else:
        df_hits  = pd.read_hdf(f, "data")
    
    # df_hits  = df_hits[df_hits.event_id.isin(event_list)]
    counter+=len(df_parts.event_id.unique())
    # df_hits_all.append(df_hits)
    
    print("Tot saved events:", len(df_parts.event_id.unique()), counter)
    
    # if counter > 900:
    #     break

# df_hits_all = pd.concat(df_hits_all)

if diffusion == "nexus":
    df_parts_all = pd.concat(df_parts_all)

# new filename: ATPC_0nubb_0_filtered.h5
if (filt_stage =="MLP"):
    # outfile = output_dir / f"{mode}_{pressure}_{diffusion}_filtered.h5"
    outfile = output_dir / f"{mode}_{pressure}_{diffusion}_filtered_highstats.h5"
else:
    outfile = output_dir / f"{mode}_{pressure}_{diffusion}_filtered_GNN_true.h5"
print(f"{f} → {outfile}")

with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    # store.put('MC/hits', df_hits_all, format='table')
    
    if diffusion == "nexus":
        store.put('MC/particles', df_parts_all, format='table')
        sys.exit(0)
    
# ------------------------------------------------------------------------------------------

Tracks = []
connections = []
connection_counts = []

# files = sorted(glob.glob(f"/media/argon/HardDrive_8TB/Krishan/ATPC//trackreco/{mode}/{pressure}/{diffusion}/pkl/*.pkl"))
files = sorted(glob.glob(f"/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples/{mode}/{pressure}/{diffusion}/pkl/*.pkl"))

Tracks = {}
connections = {}
connection_counts = {}
counter = 0
for index, f in enumerate(files):

    if index %50 ==0:
        print(f"{index} /", len(files))

    with open(f, 'rb') as pickle_file:  # Use 'rb' for reading in binary

        # Load data from file
        try:
            with open(f, 'rb') as pickle_file:
                track_data = pickle.load(pickle_file)
                conn_data = pickle.load(pickle_file)
                conn_count_data = pickle.load(pickle_file)
                
        except Exception as e:
            print(f"An unexpected error occurred while loading '{f}': {e}")
            continue

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
        
        if counter > 90000:
            break

print("Tot saved events:", counter)
if (filt_stage =="MLP"):
    # file_out = output_dir / f"{mode}_{pressure}_{diffusion}_pkl_filtered.h5"
    file_out = output_dir / f"{mode}_{pressure}_{diffusion}_pkl_filtered_highstats.h5"
else:
    file_out = output_dir / f"{mode}_{pressure}_{diffusion}_pkl_filtered_GNN.h5"
print(f"{file_out}")
with open(f"{file_out}", 'wb') as pickle_file:
    pickle.dump(Tracks, pickle_file)
    pickle.dump(connections, pickle_file)
    pickle.dump(connection_counts, pickle_file)
