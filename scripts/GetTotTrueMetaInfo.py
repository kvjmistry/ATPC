
import pandas as pd
import glob
import sys

mode="0nubb"
pressure=1
file_path = f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/ATPC_{mode}/{pressure}bar/nexus/"

files = sorted(glob.glob(f"{file_path}/*.h5"))

dfs = []
df_meta = []

for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    df_meta.append(pd.read_hdf(f, "/MC/meta"))


df_meta = pd.concat(df_meta)

cols_to_sum = ['N_gen', 'N_saved', 'N_savedE1', 'N_savedE2']
df_meta[cols_to_sum] = df_meta[cols_to_sum].astype(int) # They are saved as strings...
print(df_meta[cols_to_sum].sum())



