import glob
import pandas as pd
import pickle
import sys

mode="Bi_ion"
pressure=15
file_path = f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/ATPC_{mode}/{pressure}bar/GammaTable/"
file_out = f"files/GammaTable_{mode}_{pressure}bar_merged.h5"


files = sorted(glob.glob(f"{file_path}/*.h5"))

dfs = []
df_meta = []

for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    df_meta.append(pd.read_hdf(f, "MC/E"))


df_meta = pd.concat(df_meta)

print(df_meta)

with pd.HDFStore(f"{file_out}", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/E', df_meta, format='table')

