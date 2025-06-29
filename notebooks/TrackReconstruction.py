import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import copy
import itertools
import matplotlib.pyplot as plt
from TrackReconstruction_functions import *
import sys
import pickle
import os
import re

pd.options.mode.chained_assignment = None  # Disable the warning


# USAGE: python TrackReconstruction.py <infile> <pressure> <diffusion amount> <plot>
# python TrackReconstruction.py "ATPC_0nubb_15bar_smear_144.h5" 1 "nodiff" 0

# Input file
infile     = sys.argv[1]
print("Input file is:", infile)

# Extract the jobid from the filename
match = re.search(r"_(\d+)\.h5", infile)

if match:
    jobid = int(match.group(1))
    print("JOBID: ", jobid) 
else:
    print("Error could not find jobid, setting to 999999")
    jobid = 999999


pressure = int(sys.argv[2])
print("Pressure:", pressure, "bar")
diffusion= sys.argv[3]
print("diffusion:",diffusion)

if (diffusion != "nodiff"):
    print("Including Clustering!")
    cluster = 1
else:
    print("No Clustering!")
    cluster = 0


file_out_seg = os.path.basename(infile.rsplit('.', 1)[0])
plot=int(sys.argv[4])
print("Plotting mode:", plot)

hits = pd.read_hdf(infile,"MC/hits")
# parts = pd.read_hdf(infile,"MC/particles")

Track_dict = {}
connected_nodes_dict = {}
connections_count_dict = {}
df_list = []
df_meta = []

print("Total events to process:", len(hits.event_id.unique()))

for index, event_num in enumerate(hits.event_id.unique()):
    print("On index, Event:", index, event_num)

    # if (index > 2):
    #     continue

    hit = hits[hits.event_id == event_num]

    # These different function calls allow for different sorting if there is soeme kind of failure
    df, Tracks, connected_nodes, connection_count, pass_flag, contained     = RunTracking(hit, cluster, pressure, diffusion, 0)
    if (not pass_flag):
        print("Error in track reco, try resorting hits")
        df, Tracks, connected_nodes, connection_count, pass_flag, contained = RunTracking(hit, cluster, pressure, diffusion, 1)
    if (not pass_flag):
        print("Error in track reco, try resorting hits")
        df, Tracks, connected_nodes, connection_count, pass_flag, contained = RunTracking(hit, cluster, pressure, diffusion, 2)

    if (not pass_flag):
        print("Track still failed, skipping,...")
        continue

    Track_dict[event_num]             = Tracks
    connected_nodes_dict[event_num]   = connected_nodes
    connections_count_dict[event_num] = connection_count
    df_list.append(df)
    temp_meta = GetTrackdf(df, Tracks, 500/pressure, 180/pressure, 200/pressure, pressure) # scale these params inversely with the pressure
    temp_meta = UpdateTrackMeta(temp_meta, df, 100/pressure) # Merge deltas and brems that are near the blobs in the metadata
    temp_meta["contained"] = contained
    df_meta.append(temp_meta)

    print("Printing Metadata\n", temp_meta)
    print("\n\n")


df = pd.concat(df_list)
df_meta = pd.concat(df_meta)

with pd.HDFStore(f"{file_out_seg}_reco.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('data', df, format='table')
    store.put('meta', df_meta, format='table')


with open(f"{file_out_seg}_trackinfo.pkl", 'wb') as pickle_file:
    pickle.dump(Track_dict, pickle_file)
    pickle.dump(connected_nodes_dict, pickle_file)
    pickle.dump(connections_count_dict, pickle_file)


if plot:
    print("Plotting Events")
    for index, evt in enumerate(df.event_id.unique()):

        print("On index, Event:", index, evt)
        # if (evt != 11300):
        #     continue

        temp_df = df[df.event_id == evt]
        # temp_df = temp_df.sort_values(by='id')
        temp_df.index = temp_df.id

        # print(temp_df)

        connected_nodes = connected_nodes_dict[evt]
        connection_count = connections_count_dict[evt]
        Tracks = Track_dict[evt]

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # Plot xy projection
        plot_tracks(axs[0], temp_df['x'], temp_df['y'], connection_count, 'X', 'Y', Tracks)

        # Plot xz projection
        plot_tracks(axs[1], temp_df['x'], temp_df['z'], connection_count, 'X', 'Z', Tracks)

        # Plot yz projection
        plot_tracks(axs[2], temp_df['y'], temp_df['z'], connection_count, 'Y', 'Z', Tracks)

        plt.tight_layout()
        dir_path = f"plots/TrackingAlgoOut/{pressure}bar/{diffusion}"
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(f"plots/TrackingAlgoOut/{pressure}bar/{diffusion}/event_{evt}.pdf")


        plt.close()