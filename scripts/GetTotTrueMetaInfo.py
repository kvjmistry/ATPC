import pandas as pd
import glob
import sys

mode="Bi_ion"
pressure=int(sys.argv[1])
thick=int(sys.argv[2])
file_path = f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/ATPC_{mode}/{pressure}bar/nexus/"
file_path=f"/media/argon/HardDrive_8TB/Krishan/ATPC/GammaThickness/{mode}/{thick}cm/{pressure}bar/nat/*/"
#file_path = f"/media/argon/HardDrive_8TB/Krishan/ATPC/NEXT1t/Bi_ion/INNER_SHIELDING/*/"
#file_path = f"/media/argon/HardDrive_8TB/Krishan/ATPC/PTFE/{mode}/*/"
print(file_path)

print("Pressure:", pressure)
print("Thickness: ", thick)


files = sorted(glob.glob(f"{file_path}/*Efilt.h5"))

dfs = []
df_meta = []

for i, f in enumerate(files):
    if i %50 ==0:
        print(f"{i} /", len(files))

    df_meta.append(pd.read_hdf(f, "/MC/meta"))


df_meta = pd.concat(df_meta)

cols_to_sum = ['N_gen', 'N_saved', 'N_savedE1', 'N_savedE2', 'N_savedE1C']
#cols_to_sum = ['N_gen', 'N_saved', 'N_savedE1', 'N_savedE2', 'N_savedE3', 'N_savedE1C']
df_meta[cols_to_sum] = df_meta[cols_to_sum].astype(int) # They are saved as strings...
print(df_meta[cols_to_sum].sum())
print("\n")