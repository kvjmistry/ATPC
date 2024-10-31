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


def RunReco(data, cluster):

    # There seems to be a duplicate row sometimes
    data = data.drop_duplicates()

    # display(data)
    # eid = data.event_id.item()
    data = data[['event_id', 'x', 'y', 'z',"energy"]]

    # shuffle the data to ensure we dont use g4 ordering
    data = data.sample(frac=1).reset_index(drop=True)

    # Cluster the data if required
    if (cluster):
       data =  RunClustering(data, [10,20,30], 30)

    # then sort it based on the x,y,z
    data = data.sort_values(by=['x', "y", "z"]).reset_index(drop=True)

    # Calculate the distance matrix
    dist_matrix = distance_matrix(data[['x', 'y', 'z']], data[['x', 'y', 'z']])

    # Initialize connections counter, keeps track of number of connections to each index
    connection_count = np.zeros(len(data), dtype=int)

    # This is a dict, format is
    # index : [connected node 1, connected node 2,...]
    connected_nodes = {}
    connections = []

    # Tunable parameters
    # init_dist_thresh = 15 # max distance for initial connections [mm]
    # incr_dist_thresh = [2,4,6,8,10,12,14,16,18,20] # Second stage, look for closest nodes, then slowly increase threshold [mm]

    init_dist_thresh = 35 # max distance for initial connections [mm]
    incr_dist_thresh = [2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50] # Second stage, look for closest nodes, then slowly increase threshold [mm]


    for i in range(len(data)):
        # Find the index of the closest node (excluding itself)
        # closest_idx = np.argpartition(dist_matrix[i], 1)[1]
        closest_idx = np.argsort(dist_matrix[i])[1]
        
        # Check if the connection already exists 
        if closest_idx not in connected_nodes.get(i, []) and i not in connected_nodes.get(closest_idx, []):

            # Check the proposed node has 0 or 1 connection
            if (connection_count[closest_idx] <= 1 and connection_count[i] <= 1 and dist_matrix[i][closest_idx] < init_dist_thresh):
                
                cycle  = Testcycle(i, closest_idx ,connected_nodes, connections, connection_count)
                
                # Add connection between node i and closest_idx if it doesnt form a cycle
                if (not cycle):
                    UpdateConnections(i, closest_idx, connected_nodes, connections, connection_count)

    # Get indices where the value is 1
    single_nodes = np.where(connection_count == 1)[0]

    # Incrementally loop over distance steps looking for connections
    # starting from a small step size helps lock onto the nearest nodes
    for dist in incr_dist_thresh:

        # Connect single nodes to the next closest single node
        for i in single_nodes:
            
            # Connections get updated, so this ensures we dont make a connection to a newly formed connection
            if connection_count[i] == 1:
                
                # Find the index of the closest node with one connection (excluding itself)
                sorted_indices = np.argsort(dist_matrix[i])[1:]
                
                for closest_idx in sorted_indices[:dist]:

                    # Check if the index is not itelf and the connection count of the closest index is 1
                    if closest_idx != i and connection_count[closest_idx] <= 1 and connection_count[i] <= 1 and closest_idx not in connected_nodes.get(i, []) and i not in connected_nodes.get(closest_idx, []): 
                        
                        if dist_matrix[i][closest_idx] < dist:

                            cycle  = Testcycle(i, closest_idx ,connected_nodes, connections, connection_count)
                            
                            if not cycle:
                                UpdateConnections(i, closest_idx, connected_nodes, connections, connection_count)
                                break

    # Get indices where the value is 1
    single_nodes = np.where(connection_count == 1)[0]

    Tracks = []

    for i,node in enumerate(single_nodes):
        # Check that the track hasnt already been added
        if (check_start_end_exists(node,Tracks)):
            continue

        # Get the track path
        path = GetNodePath(connected_nodes, node, connected_nodes[node][0])

        total_length, total_energy = GetTrackLengthEnergy(path, data)
        color = next(color_cycle)

        Track = {"id":i, "start":path[0], "end":path[-1], "nodes":path, "length":total_length, "energy":total_energy,"label":"track","c":color}
        Tracks.append(Track)


    # for t in Tracks:
    #     print(t)

    # print(GetMeanNodeDist(Tracks, data))

    dist_threshold = 4*GetMeanNodeDist(Tracks, data)

    # Add in any nodes without connections to the tracks as gammas and re-label other tracks as gammas
    AddConnectionlessNodes(connection_count, Tracks, data)

    finished = False  # Initial state

    q = 0
    while not finished:
        # print("Loop: ", q)

        finished, Tracks = ConnectTracks(Tracks, connected_nodes, connections, connection_count, dist_matrix, dist_threshold, data)

        q=q+1


    print("Total Tracks:", len(Tracks))


    ## Set the colours of the tracks
    Tracks = CategorizeTracks(Tracks)


    num_nodes = 0
    primary_track_id = -1
    primary_nodes = []

    for t in Tracks:
        if ( len(t["nodes"]) > num_nodes):
            num_nodes = len(t["nodes"])
            primary_track_id = t["id"]
            primary_nodes = t["nodes"]

    print("The primary track is: ", primary_track_id)

    # This list makes sure we only have one angle per node
    all_visited = []

    Primary_Track = data.reindex(primary_nodes)
    Primary_Track = GetAnglesDF(Primary_Track, all_visited, 1, primary_track_id)
    all_visited = all_visited + primary_nodes

    df_angles = pd.DataFrame()
    df_angles = pd.concat([df_angles, Primary_Track], ignore_index=True)

    for t in Tracks:

        trk_nodes = t["nodes"]
        if t["id"] == primary_track_id:
            continue
        else:
            
            # See if the first/last node has three connections. If it does then flip the track
            con_end = connection_count[t["nodes"][-1]]
            if (con_end == 3):
                trk_nodes = trk_nodes[::-1]
            
            trk = data.reindex(trk_nodes)

            trk = GetAnglesDF(trk, all_visited, 0, t["id"])
            all_visited = all_visited + trk_nodes
            df_angles = pd.concat([df_angles, trk], ignore_index=True)


    print(df_angles)
    return df_angles, Tracks, connected_nodes, connection_count


# USAGE: python TrackReconstruction.py <infile> <clustering option>
# python TrackReconstruction.py "Leptoquark_SM_nexus.h5" 0

# Input file
infile     = sys.argv[1]
cluster = int(sys.argv[2])
jobid   = int(sys.argv[3])
file_out_seg = os.path.basename(infile.rsplit('.', 1)[0])
plot=True

hits = pd.read_hdf(infile,"MC/hits")
# parts = pd.read_hdf(infile,"MC/particles")

Track_dict = {}
connected_nodes_dict = {}
connections_count_dict = {}
df_list = []

for event_num in hits.event_id.unique():
    print("On Event:", event_num)

    hit = hits[hits.event_id == event_num]

    df, Tracks, connected_nodes, connection_count = RunReco(hit, cluster)
    Track_dict[event_num] = Tracks
    connected_nodes_dict[event_num] = connected_nodes
    connections_count_dict[event_num] = connection_count
    df_list.append(df)



df = pd.concat(df_list)

# df.to_hdf(f"{file_out_seg}_{jobid}_reco.h5", key='data', mode='w')


# with open(f"{file_out_seg}_{jobid}_trackinfo.pkl", 'wb') as pickle_file:
#     pickle.dump(Track_dict, pickle_file)
#     pickle.dump(connected_nodes_dict, pickle_file)
#     pickle.dump(connections_count_dict, pickle_file)

if plot:
    for evt in df.event_id.unique():

        # if (evt == 501):
        #     break

        temp_df = df[df.event_id == evt]
        # temp_df = temp_df.sort_values(by='id')
        temp_df.index = temp_df.id

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
        plt.savefig(f"plots/TrackingAlgoOut/event_{evt}.pdf")




# df_reco = pd.DataFrame(mydict_reco) 
# df_reco.to_csv(file_out, sep=',', index=False, header=False) 