import pandas as pd
import glob


pressures = [1,5,10,15]
modes = ["0nubb", "Bi", "Tl"]



for m in modes:

    print("On Mode:", m)

    dfs = []

    for p in pressures:
        print("On Pressure:", p)

        files = sorted(glob.glob(f"/media/argon/HardDrive_8TB/Krishan/ATPC/NEXUSTRUE/${m}/${p}bar/*.h5"))

        for f in files:
            df = pd.read_hdf(f, "trueinfo")
            dfs.append(df)

    dfs = pd.concat(dfs)

    print(dfs)

    with pd.HDFStore(f"NEXUS_True_ATPC_{m}", mode='w', complevel=5, complib='zlib') as store:
        store.put('trueinfo', dfs, format='table')








