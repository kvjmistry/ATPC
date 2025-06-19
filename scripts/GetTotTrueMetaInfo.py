
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

    df_meta.append(pd.read_hdf(f, "/MC/meta))


df_meta = pd.concat(df_meta)

print(df_meta.sum(numeric_only=True))



