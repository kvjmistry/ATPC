import glob
import pandas as pd
import sys
from pathlib import Path

# Example
# python3 FilterEventsEventIDMerge.py ATPC_Bi_ion 1bar 5percent

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

    df_hits_all.appen(df_hits)
    print("Tot saved events:", len(df_hits.event_id.unique()), counter)

df_hits_all = pd.concat(df_hits_all)

# new filename: ATPC_0nubb_0_filtered.h5
outfile = output_dir / f"{mode}_{pressure}_{diffusion}_filtered.h5"
print(f"{f} → {outfile}")

with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/hits', df_hits_all, format='table')
