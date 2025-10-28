import glob
import pandas as pd
import pickle
import sys

mode="Tl_ion"
pressure=25
thick="4cm"
volume="FIELD_CAGE"
#file_path = f"/media/argon/HardDrive_8TB/Krishan/ATPC/GammaThickness/{mode}/{thick}/25bar/enr/*/"
#file_out = f"merged/GammaTable_{mode}_{pressure}bar_{thick}_merged.h5"
file_path = f"/media/argon/HardDrive_8TB/Krishan/ATPC/NEXT1t/{mode}/{volume}/*"
file_out = f"merged/NextTonne/GammaTable_{mode}_{volume}_merged.h5"

#file_path = f"/media/argon/HardDrive_8TB/Krishan/ATPC/PTFE/{mode}/*/"
#file_out = f"merged/NextTonne/GammaTable_ATPC_PTFE_{mode}_merged.h5"


files = sorted(glob.glob(f"{file_path}/*gamma*.h5"))

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

