import pandas as pd
import glob


pressures = [1,5,10,15,25]
modes = ["0nubb", "single"]



for m in modes:

    print("On Mode:", m)

    dfs = []

    for p in pressures:
        print("On Pressure:", p)

        files = sorted(glob.glob(f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/ATPC_{m}/{p}bar/TrueInfo/*.h5"))

        for f in files:
            df = pd.read_hdf(f, "trueinfo")
            dfs.append(df)

    dfs = pd.concat(dfs)

    print(dfs)

    with pd.HDFStore(f"files/TrueInfo_ATPC_{m}.h5", mode='w', complevel=5, complib='zlib') as store:
        store.put('trueinfo', dfs, format='table')








