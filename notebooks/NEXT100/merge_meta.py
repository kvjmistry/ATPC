import glob
import pandas as pd
import pickle
import sys

# Example

# python3 merge_outputs.py BLOBR

# Set this to true to save the data info to file not just metadata
save_datainfo = False

blobr=sys.argv[1]

print("blobr:", blobr)

run_num = 354015
run_num = 230725

file_path = f"/media/argon/HardDrive_8TB/Krishan/ATPC/{run_num}/TrackReco/blobR_{blobr}/"
file_out = f"./merged/"

print("file_path", file_path)
print("file_out", file_out)

files = sorted(glob.glob(f"{file_path}/*.h5"))

df_meta = []

for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    # dfs.append(pd.read_hdf(f, "data"))
    df_meta.append(pd.read_hdf(f, "meta"))

df_meta = pd.concat(df_meta)


print(df_meta)

with pd.HDFStore(f"{file_out}/trackreco_sophronia_{run_num}_blobr{blobr}_reco.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('meta', df_meta, format='table')
