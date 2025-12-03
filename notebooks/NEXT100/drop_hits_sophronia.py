'''
A script to drop ioslated hits for sophronia
'''

import numpy as np
import pandas as pd
import os
import tables as tb
import glob
import sys
os.environ["ICTDIR"] = "~/Packages/IC"
os.environ["ICDIR"] = "$ICTDIR/invisible_cities"

from invisible_cities.cities.beersheba import drop_isolated
from invisible_cities.io.dst_io        import df_writer
from invisible_cities.io.dst_io import load_dst



def read_data(input, run_number):
    '''
    takes in and reads out all the required beersheba tables
    '''
    dsts = {}

    dsts["RECO"]    = pd.read_hdf(input, "RECO/Events")
    dsts['DST']     = pd.read_hdf(input, 'DST/Events')
    dsts['runevts'] = pd.read_hdf(input, 'Run/events')
    dsts['runinfo'] = pd.read_hdf(input, 'Run/runInfo')


   # read out all this madness from a dictionary
    return dsts


def sophronia_hits(infile, output_directory, run_number, drop_dist = [16, 16, 4], nhits = 3):

    drop_sensors = drop_isolated(drop_dist, ['Ec'], nhits)
    # extract all files in the input directory
    #files = [f for f in os.listdir(input_directory) if f.endswith('.h5')]
    #files = glob.glob(f"{input_directory}/*.h5")
    files = [infile]
    print("Files:", files)
    for f in files:
        basename = os.path.basename(f)
        basename2 = os.path.splitext(basename)[0]
        output_name = f'{basename2}_dh.h5'
        print("Outputname is:", output_name)
        dsts = read_data(f'{f}', run_number)
        print(dsts["RECO"])

        file_data = []
        for i, df in dsts['RECO'].groupby('event'):

            print(f'event {i}:')

            # drop isolated sensors
            try:
                dropped_df = drop_sensors(df.copy())
            except Exception as e:
                print(f"Error while dropping sensors: {e}")
                dropped_df = df.copy()  # fallback to original DataFrame if needed

            file_data.append(dropped_df)

        new_df = pd.concat(file_data)

        print(new_df)

        # save
        print("Saving data to:", f'{output_directory}/{output_name}')
        with pd.HDFStore(f'{output_directory}/{output_name}', mode='w', complevel=5, complib='zlib') as store:
            store.put('RECO/Events',new_df,    format='table')

infile=sys.argv[1]

# Load in the data
# sophronia_hits(infile, "/media/argon/HardDrive_8TB/Krishan/ATPC/354015/rebinned/", 99)
sophronia_hits(infile, "/media/argon/HardDrive_8TB/Krishan/ATPC/230725/droppedhits/", 99)
# sophronia_hits(infile, "./data/", 99)


