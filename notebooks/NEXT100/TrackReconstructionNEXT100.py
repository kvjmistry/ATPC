import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import copy
import itertools
import matplotlib.pyplot as plt
import sys
import pickle
import os
import re

sys.path.append("../../scripts")
from TrackReconstruction_functions import *

pd.options.mode.chained_assignment = None  # Disable the warning


# USAGE: python TrackReconstructionNEXT100.py <infile> <distance_threshold> <radius_threshold> <plot>
# python TrackReconstructionNEXT100.py "run_15589_5302_ldc5_230725_beersheba.h5" 100 40 0

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


distance_threshold = float(sys.argv[2])
print("Distance Threshold:", distance_threshold, "mm")
radius_threshold= float(sys.argv[3])
print("Radius Threshold:", radius_threshold, "mm")

pressure = 5
diffusion = "deconv"

# if (diffusion != "nodiff"):
#     print("Including Clustering!")
#     cluster = 1
# else:
#     print("No Clustering!")
#     cluster = 0

cluster = 1


file_out_seg = os.path.basename(infile.rsplit('.', 1)[0])
plot=int(sys.argv[4])
print("Plotting mode:", plot)

hits = pd.read_hdf(infile,"DECO/Events")
hits = hits[["event", "X", "Y", "Z", "E"]]
hits = hits.rename(columns={"event":"event_id","X": "x", "Y": "y", "Z": "z", "E":"energy"})
# parts = pd.read_hdf(infile,"MC/particles")

Track_dict = {}
connected_nodes_dict = {}
connections_count_dict = {}
df_list = []
df_meta = []

print("Total events to process:", len(hits.event_id.unique()))

for index, event_num in enumerate(hits.event_id.unique()):
    print("On index, Event:", index, event_num)

    # if (index > 5):
    #     break

    hit = hits[hits.event_id == event_num]

    # These different function calls allow for different sorting if there is soeme kind of failure
    df, Tracks, connected_nodes, connection_count, pass_flag, contained     = RunTracking(hit, cluster, pressure, diffusion, 0)
    if (not pass_flag):
        print("Error in track reco, try resorting hits\n")
        df, Tracks, connected_nodes, connection_count, pass_flag, contained = RunTracking(hit, cluster, pressure, diffusion, 1)
    if (not pass_flag):
        print("Error in track reco, try resorting hits\n")
        df, Tracks, connected_nodes, connection_count, pass_flag, contained = RunTracking(hit, cluster, pressure, diffusion, 2)

    if (not pass_flag):
        print("Track still failed, skipping,...")
        continue

    Track_dict[event_num]             = Tracks
    connected_nodes_dict[event_num]   = connected_nodes
    connections_count_dict[event_num] = connection_count
    df_list.append(df)
    
    # Slightly different input params for next1t analysis
    if (diffusion == "next1t"):
        temp_meta = GetTrackdf(df, Tracks, 30, 15, 15, pressure)
    elif (diffusion == "deconv"):
        temp_meta = GetTrackdf(df, Tracks, distance_threshold, radius_threshold, radius_threshold, pressure)
    else:
        temp_meta = GetTrackdf(df, Tracks, 500/pressure, 225/pressure, 225/pressure, pressure) # scale these params inversely with the pressure
    
    
    # temp_meta = UpdateTrackMeta(temp_meta, df, 10/pressure) # Merge deltas and brems that are near the blobs in the metadata
    temp_meta = UpdateTrackMeta2(temp_meta) # ensure variables are organized so that var 1 > var 2 e.g blob1>blob2
    temp_meta["contained"] = contained
    df_meta.append(temp_meta)

    print("Printing Metadata\n", temp_meta[["event_id", "primary", "length", "energy", "blob1", "blob2", "blob1R", "blob2R", "Tortuosity1", "Tortuosity2", "Squiglicity1", "Squiglicity2", "label", "contained"]])
    print(temp_meta[["event_id", "blob1RTD", "blob2RTD"]])
    print("\n\n")


df = pd.concat(df_list)
df_meta = pd.concat(df_meta)

# Print the reconstruction efficiency and any events that failed
Reco_eff = 100*len(df_meta.event_id.unique())/ len(hits.event_id.unique())
print("Track reconstruction efficiency:", Reco_eff)
if Reco_eff < 100:
    missing_events = set(hits.event_id.unique()) - set(df_meta.event_id.unique())
    print("Events in hits but not in df_meta:", missing_events)

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

        hits_event = hits[hits.event_id == evt]

        # print(temp_df)

        connected_nodes = connected_nodes_dict[evt]
        connection_count = connections_count_dict[evt]
        Tracks = Track_dict[evt]

        # Create subplots
        fig = plt.figure(figsize=(10, 10))

        axs = [fig.add_subplot(2, 2, i + 1) for i in range(3)]

        # Plot xy projection
        plot_tracks(axs[0], temp_df['x'], temp_df['y'], connection_count, 'X', 'Y', Tracks)
        axs[0].scatter(hits_event.x, hits_event.y, c="k",s=2)

        # Plot xz projection
        plot_tracks(axs[1], temp_df['x'], temp_df['z'], connection_count, 'X', 'Z', Tracks)
        axs[1].scatter(hits_event.x, hits_event.z, c="k",s=2)

        # Plot yz projection
        plot_tracks(axs[2], temp_df['y'], temp_df['z'], connection_count, 'Y', 'Z', Tracks)
        axs[2].scatter(hits_event.y, hits_event.z, c="k",s=2)

        ax_3D = fig.add_subplot(2, 2, 4, projection='3d')
        plot_tracks_3D(ax_3D, temp_df['x'], temp_df['y'], temp_df['z'], connection_count, Tracks)
        ax_3D.scatter(hits_event.x, hits_event.y, hits_event.z, c=hits_event.energy, marker='o', alpha=0.15,s=3)

        # Get bounds of primary track 
        df_prim = temp_df[temp_df.primary == 1]

        lims = {
            "x": (df_prim.x.min()-20, df_prim.x.max()+20),
            "y": (df_prim.y.min()-20, df_prim.y.max()+20),
            "z": (df_prim.z.min()-20, df_prim.z.max()+20)
        }

        axs[0].set_xlim(*lims["x"])
        axs[0].set_ylim(*lims["y"])
        axs[1].set_xlim(*lims["x"])
        axs[1].set_ylim(*lims["z"])
        axs[2].set_xlim(*lims["y"])
        axs[2].set_ylim(*lims["z"])
        ax_3D.set_xlim(*lims["x"])
        ax_3D.set_ylim(*lims["y"])
        ax_3D.set_zlim(*lims["z"])

        plt.tight_layout()
        dir_path = f"plots/TrackingAlgoOut/"
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(f"plots/TrackingAlgoOut/event_{evt}.pdf")


        plt.close()