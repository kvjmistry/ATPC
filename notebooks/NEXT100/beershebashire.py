'''
beershebashire - a crappy adage to beersheba that:
- rebins the data so that there aren't a stupid number of interpolated hits
- drops isolated hits

This is basically a 'soft' voxelisation before paolina functions are applied, but its a necessity for working with hits directly.

'''

import numpy as np
import pandas as pd
import os
import tables as tb
os.environ["ICTDIR"] = "~/Packages/IC"
os.environ["ICDIR"] = "$ICTDIR/invisible_cities"

from invisible_cities.cities.beersheba import drop_isolated
from invisible_cities.io.dst_io        import df_writer
from invisible_cities.io.dst_io import load_dst


def rebin(df, dx, dy, dz):
    '''
    Docstring for rebin

    :param df: Description
    :param dx: Description
    :param dy: Description
    :param dz: Description
    '''


    x_edge = np.arange(df.X.min(), df.X.max() + dx, dx)
    y_edge = np.arange(df.Y.min(), df.Y.max() + dy, dy)
    z_edge = np.arange(df.Z.min(), df.Z.max() + dz, dz)

    He, edges = np.histogramdd(
                            sample = np.vstack([df.X, df.Y, df.Z]).T,
                            bins   = [x_edge, y_edge, z_edge],
                            weights = df.E)

    # Compute bin centers
    x_centers = 0.5 * (edges[0][1:] + edges[0][:-1])
    y_centers = 0.5 * (edges[1][1:] + edges[1][:-1])
    z_centers = 0.5 * (edges[2][1:] + edges[2][:-1])

    # Make meshgrid of centers
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    rebin_df = pd.DataFrame({
                     "event": df.event.unique()[0],
                     "npeak": df.npeak.unique()[0],
                     "X"        : xx.ravel(),
                     "Y"        : yy.ravel(),
                     "Z"        : zz.ravel(),
                     "E"        : He.ravel()})
    
    # remove empty hits (you need to do this)
    rebin_df = rebin_df[rebin_df['E'] > 0].reset_index(drop=True)


    return rebin_df


def read_data(input, run_number):
    '''
    takes in and reads out all the required beersheba tables
    '''
    dsts = {}

    dsts["DECO"]    = pd.read_hdf(input, "DECO/Events")
    dsts['DST']     = pd.read_hdf(input, 'DST/Events')
    dsts['filters'] = pd.read_hdf(input, 'Filters/nohits')
    dsts['runevts'] = pd.read_hdf(input, 'Run/events')
    dsts['runinfo'] = pd.read_hdf(input, 'Run/runInfo')
    # dsts['conf']    = pd.read_hdf(input, 'config',  'beersheba')

    # load in MC if labelled as an MC run
    # if int(run_number) <= 0:
    #     dsts['MC_conf']      = load_dst(input, 'MC', 'configuration')
    #     dsts['MC_evmp']      = load_dst(input, 'MC', 'event_mapping')
    #     dsts['MC_hits']      = load_dst(input, 'MC', 'hits')
    #     dsts['MC_particles'] = load_dst(input, 'MC', 'particles')
    #     dsts['MC_snspos']    = load_dst(input, 'MC', 'sns_positions')
    #     dsts['MC_sns_resp']  = load_dst(input, 'MC', 'sns_response')
    #     dsts['MC_evtmp']     = load_dst(input, 'Run', 'eventMap')

   # read out all this madness from a dictionary
    return dsts


def save_data(dsts, rebin_df, output_directory, save_file_name, run_number):
    '''
    saves individual files
    '''

    print(f'Saving data to {output_directory}{save_file_name}')

    with tb.open_file(f'{output_directory}{save_file_name}') as h5out:
        df_writer(h5out, dsts['DST'],      'DST',     'Events')
        # df_writer(h5out, rebin_df,         'DECO',    'Events')
        # df_writer(h5out, dsts['filters'],  'Filters', 'nohits')
        # df_writer(h5out, dsts['runevts'],  'Run',     'events')
        # df_writer(h5out, dsts['runinfo'],  'Run',     'runInfo')
        # df_writer(h5out, dsts['conf'],     'config',  'beersheba')

        # if int(run_number) <= 0:
        #     df_writer(h5out, dsts['MC_conf'],          'MC',  'configuration')
        #     df_writer(h5out, dsts['MC_evmp'],          'MC',  'event_mapping')
        #     df_writer(h5out, dsts['MC_hits'],          'MC',  'hits')
        #     df_writer(h5out, dsts['MC_particles'],     'MC',  'particles')
        #     df_writer(h5out, dsts['MC_snspos'],        'MC',  'sns_positions')
        #     df_writer(h5out, dsts['MC_sns_resp'],      'MC',  'sns_response')
        #     df_writer(h5out, dsts['MC_evtmp'],         'MC',  'eventMap')


def beershireba(input_directory, output_directory, run_number, rebin_d = [5,5,4], drop_dist = [16, 16, 4], nhits = 3):

    drop_sensors = drop_isolated(drop_dist, ['E'], nhits)
    # extract all files in the input directory
    files = [f for f in os.listdir(input_directory) if f.endswith('.h5')]
    print("Files:", files)
    for f in files:
        basename = os.path.basename(f)
        basename2 = os.path.splitext(basename)[0]
        output_name = f'{basename2}_beershireba.h5'
        print("Outputname is:", output_name)
        dsts = read_data(f'{input_directory}{f}', run_number)
        print(dsts["DECO"])

        file_data = []
        for i, df in dsts['DECO'].groupby('event'):

            print(f'event {i}:')

            # rebin
            rebinned_df = rebin(df, rebin_d[0], rebin_d[1], rebin_d[2])

            # drop isolated sensors
            dropped_df  = drop_sensors(rebinned_df.copy())

            file_data.append(dropped_df)

        new_df = pd.concat(file_data)

        # save
        print("Saving data")
        save_data(dsts, new_df, output_directory, output_name, run_number)


# Load in the data
beershireba("data/", "data/", 99)

