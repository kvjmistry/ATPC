import glob
import pandas as pd
import pickle
import sys

# Example
# python3 FilterEvents.py ATPC_Bi 1bar 5percent nexus


mode=sys.argv[1]
pressure=sys.argv[2]
diffusion=sys.argv[3]
true_folder=sys.argv[4] # can also set 5percent here to get the diffused track

file_path = f"/home/krishan.mistry/code/ATPC/merged/{mode}_{pressure}_{diffusion}_reco.h5"
file_out = f"/home/krishan.mistry/code/ATPC/merged/{mode}_{pressure}_{true_folder}_filtered"

hits = pd.read_hdf(f"{file_path}", "data")
filtered_events = hits.event_id.unique()

print("Total events to filter:", len(filtered_events))

# ------------------------------------------------------------------------------------------

# Now load in the nexus files
files = sorted(glob.glob(f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/{mode}/{pressure}/{true_folder}/*.h5"))

df_hits_all = []
df_parts_all = []

counter = 0
for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    df_hits  = pd.read_hdf(f, "MC/hits")
    df_parts = pd.read_hdf(f, "MC/particles")
    df_hits  = df_hits[df_hits.event_id.isin(filtered_events)]
    df_parts = df_parts[df_parts.event_id.isin(filtered_events)]
    counter+=len(df_hits.event_id.unique())
    
    if (len(df_hits) > 0):
        df_hits_all.append(df_hits)
        df_parts_all.append(df_parts)

    # If we got all the events already then we can stop here!
    if (counter >= len(filtered_events)):
        break

df_hits_all  = pd.concat(df_hits_all)
df_parts_all = pd.concat(df_parts_all)

print(df_hits_all)
print(df_parts_all)
print("Tot saved events:", len(df_parts_all.event_id.unique()))

with pd.HDFStore(f"{file_out}.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/hits', df_hits_all, format='table')
    store.put('MC/particles', df_parts_all, format='table')
