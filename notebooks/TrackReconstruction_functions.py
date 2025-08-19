import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import itertools
import copy
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

colormap = plt.cm.get_cmap('Dark2')
color_cycle = itertools.cycle(colormap.colors)


# ---------------------------------------------------------------------------------------------------
# Function to add connections made to the dictionaries
# curr_node_indx   : the node we are currently on
# conn_node_idx    : the proposed node to connect to
# connected_nodes_ : dictionary that keeps track of what is connected to each node
#                    format is: index : [connected node 1, connected node 2,...]
# connections_     :  a simple list that contains pairs of connections e.g. [ (1,3), (3,5)),...]
# connection_count_: list of the number of connections to each node
def UpdateConnections(curr_node_idx, conn_node_idx, connected_nodes_, connections_, connection_count_):

    # We shouldnt be doing any self connection
    if (curr_node_idx == conn_node_idx):
        return

    # Add connection between node i and closest_idx
    connections_.append((curr_node_idx, conn_node_idx))
    connection_count_[curr_node_idx] += 1
    connection_count_[conn_node_idx] += 1
    
    # Update connected nodes
    if curr_node_idx in connected_nodes_:
        connected_nodes_[curr_node_idx].append(conn_node_idx)
    else:
        connected_nodes_[curr_node_idx] = [conn_node_idx]
    
    if conn_node_idx in connected_nodes_:
        connected_nodes_[conn_node_idx].append(curr_node_idx)
    else:
        connected_nodes_[conn_node_idx] = [curr_node_idx]

    return connected_nodes_, connections_, connection_count_

# ---------------------------------------------------------------------------------------------------
# Function to check if a new connection would form a closed loop
# this helps to build the track as a continous object without loops
# node            : the start node
# target          : the proposed node to check if we connected will form a loop
# connections_dict: same as connected_nodes, 
#                   a dictionary that keeps track of what is connected to each node
#                    format is: index : [connected node 1, connected node 2,...]
def forms_cycle(node, target, connections_dict):

    query = node
    prev_node = node 
    # print(query)

    # Loop
    for index,n in enumerate(range(len(connections_dict))):
        
        # Get the nodes connected to the present node we are checking
        con_nodes = connections_dict[query]
        # print("Start",query, prev_node, con_nodes)

        # We hit a end-point and it didnt loop (only one node connected)
        if (len(con_nodes) == 1):
            return False

        # Get the node that went in the query before
        if con_nodes[1] == prev_node:
            prev_node = query
            query = con_nodes[0]
        else:
            prev_node = query
            query = con_nodes[1]

        if (index == 0):
            query = con_nodes[0]
            prev_node = node


        # If the returned query value is the target then we have looped
        if (query == target):
            return True

    # We looped over everything and found no loops
    return False
    
# ---------------------------------------------------------------------------------------------------
# Function for testing for closed loops
# We copy so dont impact the original dicts
# We then trial the connection to see if it loops or not. Return true if cycle, else false
# curr_node: the present node we are looking to connect to
# conn_node: the proposed node to connect to
# connected_nodes_ : dictionary that keeps track of what is connected to each node
#                    format is: index : [connected node 1, connected node 2,...]
# connections_     :  a simple list that contains pairs of connections e.g. [ (1,3), (3,5)),...]
# connection_count_: list of the number of connections to each node
def Testcycle(curr_node, conn_node, connected_nodes_, connections_, connection_count_):

    # Temporarily add the connection to check for cycles
    temp_connections_dict = copy.deepcopy(connected_nodes_)
    temp_connections      = copy.deepcopy(connections_)
    temp_connection_count = copy.deepcopy(connection_count_)

    # print(i,closest_idx,connection_count[i], connection_count[closest_idx], temp_connections_dict[i], temp_connections_dict[closest_idx])
    temp_connections_dict, temp_connections, temp_connection_count = UpdateConnections(curr_node, conn_node, temp_connections_dict, temp_connections, temp_connection_count)

    # Check for cycles
    cycle = forms_cycle(curr_node, conn_node, temp_connections_dict)

    temp_connections_dict = {}
    temp_connections = []
    temp_connection_count = []

    return cycle

# ---------------------------------------------------------------------------------------------------
def check_start_end_exists(number,Tracks):
    check_start = any(path["start"] == number for path in Tracks)
    check_end = any(path["end"] == number for path in Tracks)

    if (check_start or check_end):
        return True
    else:
        return False

# ---------------------------------------------------------------------------------------------------
# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2['x'] - point1['x'])**2 + (point2['y'] - point1['y'])**2 + (point2['z'] - point1['z'])**2)

# ---------------------------------------------------------------------------------------------------
# Function to calculate the Euclidean distance between two points -- use with numpy arrays
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# ---------------------------------------------------------------------------------------------------
# Get the length and energy of a track
def GetTrackLengthEnergy(path, data):
    total_length = 0
    total_energy = 0

    # Return the hit if there is only one node
    if len(path) == 0:
        return 0,data.iloc[path[0]]['energy']

    for t in range(len(path) - 1):
        point1 = data.iloc[path[t]]
        point2 = data.iloc[path[t + 1]]
        
        distance = calculate_distance(point1, point2)
        total_length += distance
        total_energy += point1['energy']
    
    # Add in the last energy hit
    total_energy += data.iloc[path[-1]]['energy']

    return round(total_length, 3), total_energy
    # return round(total_length, 3), round(total_energy, 3)

# ---------------------------------------------------------------------------------------------------
# For aleady connected nodes, return the median distance of connections
def GetMedianNodeDist(Tracks, data):

    nodedists = []

    for track in Tracks:

        nodes = track["nodes"]
        if (len(nodes) == 1):
            continue

        for n in range(len(nodes) - 1):

            point1 = data.iloc[nodes[n]]
            point2 = data.iloc[nodes[n + 1]]
            
            distance = calculate_distance(point1, point2)
            nodedists.append(distance)

    return round(np.median(nodedists), 3)

# ---------------------------------------------------------------------------------------------------
def GetTrackwithNode(closest_idx, Tracks_):
    for t in Tracks_:
        if (closest_idx in t["nodes"]):
            return t["id"]
    # The node wasnt found anywhere...
    return -1

# ---------------------------------------------------------------------------------------------------
def GetTrackDictwithNode(closest_idx, Tracks_):
    for t in Tracks_:
        if (closest_idx in t["nodes"]):
            return t
    # The node wasnt found anywhere...
    return -1

# ---------------------------------------------------------------------------------------------------
def join_tracks(array1, array2):
    # Check if the arrays can be joined directly
    if array1[-1] == array2[0]:
        joined_array = array1 + array2[1:]
    elif array1[0] == array2[-1]:
        joined_array = array2 + array1[1:]
    # Check if reversing one of the arrays allows them to be joined
    elif array1[-1] == array2[-1]:
        array2_reversed = array2[::-1]
        joined_array = array1 + array2_reversed[1:]
    elif array1[0] == array2[0]:
        array1_reversed = array1[::-1]
        joined_array = array1_reversed + array2[1:]
    else:
        print("Error cannot join arrays", array1, array2)
        joined_array = array1 + array2  # If they can't be joined, just concatenate them

    return joined_array

# ---------------------------------------------------------------------------------------------------
def AddConnectedTracks(curr_track,conn_track, delta_path, seg1_path, seg2_path, UpdatedTracks_, data):

    # Get the ids before popping
    delta_id   = GetUniqueTrackID(UpdatedTracks_)
    primary_id =  GetUniqueTrackID(UpdatedTracks_)+1

    # Remove the old tracks from the array
    for index, t in enumerate(UpdatedTracks_):
        
        # remove the old tracks
        if (t["id"] == curr_track):
            UpdatedTracks_.pop(index)

    # Remove the old tracks from the array
    for index, t in enumerate(UpdatedTracks_):
        # remove the old tracks
        if (t["id"] == conn_track):
            UpdatedTracks_.pop(index)


    delta_len, delta_e = GetTrackLengthEnergy(delta_path, data)
    Delta = {"id":delta_id, "start":delta_path[0], "end":delta_path[-1], "nodes":delta_path, "length":delta_len, "energy":delta_e,"label":"track","c":"black"}
    UpdatedTracks_.append(Delta)
    
    joined_track_path = join_tracks(seg1_path, seg2_path)
    total_length_joined, total_energy_joined = GetTrackLengthEnergy(joined_track_path, data)
    color = next(color_cycle)
    
    Primary = {"id":primary_id, "start":joined_track_path[0], "end":joined_track_path[-1], "nodes":joined_track_path, "length":total_length_joined, "energy":total_energy_joined,"label":"track","c":"black"}
    UpdatedTracks_.append(Primary)

# ---------------------------------------------------------------------------------------------------
# Update an existing track in the updated tracks array from the merging of two tracks
def UpdateAndMergeTrack(curr_track,conn_track, newpath, UpdatedTracks_, data):

    name=""
    color = "black"
    primary_id = GetUniqueTrackID(UpdatedTracks_)
    
    for index, t in enumerate(UpdatedTracks_):
        
        # Remove the old tracks from the array
        for index, t in enumerate(UpdatedTracks_):
            
            # remove the old tracks
            if (t["id"] == curr_track):
                UpdatedTracks_.pop(index)

        # Remove the old tracks from the array
        for index, t in enumerate(UpdatedTracks_):
            # remove the old tracks
            if (t["id"] == conn_track):
                # name=t["label"]
                # color = t["c"]
                UpdatedTracks_.pop(index)

    # Add the new merged track
    length, energy = GetTrackLengthEnergy(newpath, data)
    Primary = {"id":primary_id, "start":newpath[0], "end":newpath[-1], "nodes":newpath, "length":length, "energy":energy,"label":"track","c":"black"}
    UpdatedTracks_.append(Primary)

# ---------------------------------------------------------------------------------------------------
def GetUniqueTrackID(Tracks_):

    temp_track_id = -1

    for t in Tracks_:
        if temp_track_id <= t["id"]:
            temp_track_id = t["id"]

    # print("New Track ID is:",temp_track_id+1)

    return temp_track_id+1

# ---------------------------------------------------------------------------------------------------
# Any nodes without a connection can be re-added as a track
def AddConnectionlessNodes(connection_count, UpdatedTracks, data):

    for index, c in enumerate(connection_count):

        if (c == 0):
            hit_energy = data.iloc[index].energy
            Gamma = {"id":GetUniqueTrackID(UpdatedTracks), "start":index, "end":index, "nodes":[index], "length":0, "energy":hit_energy,"label":"gamma","c":"y"}
            UpdatedTracks.append(Gamma)

# ---------------------------------------------------------------------------------------------------
def ConnectTracks(Tracks_, connected_nodes_, connections_, connection_count_, dist_matrix, dist_threshold, data):


    # Dont run this if we only got one track!
    if (len(Tracks_) == 1):
        return True, Tracks_, connected_nodes_, connections_, connection_count_

    for idx, Track in enumerate(Tracks_):

        # Current track
        curr_track = Track["id"]
        
        start_node = Track["start"]
        end_node   = Track["end"]

        # Get the indexes of closest nodes to start and end
        dist_ind_start = np.argsort(dist_matrix[start_node])[1:]
        dist_ind_end   = np.argsort(dist_matrix[end_node])[1:]

        # Filter nodes that are in the current track
        dist_ind_start = [x for x in dist_ind_start if x not in Track["nodes"]]
        dist_ind_end   = [x for x in dist_ind_end if x not in Track["nodes"]]

        # Distances of the end point to the closest track
        dist_start = dist_matrix[start_node][dist_ind_start[0]]
        dist_end   = dist_matrix[end_node][dist_ind_end[0]]

        # Apply threshold. If both nodes fail here then no need to continue
        if (dist_start > dist_threshold and dist_end > dist_threshold):
            # print("Failed distance requirements")
            continue

        # First find the node with the closest track
        closest_idx = 0
        end_conn_node = 0
        con_point = "start"
        curr_track_path = Track["nodes"]

        # Choose the smallest index
        if dist_start < dist_end:
            closest_idx = dist_ind_start[0]
            end_conn_node = start_node

            if (dist_start > dist_threshold):
                # print(" Start Failed distance requirements")
                continue
            
        else:
            closest_idx = dist_ind_end[0]
            end_conn_node = end_node
            con_point = "end"

            # Apply threshold
            if (dist_end > dist_threshold):
                # print(" End Failed distance requirements")
                continue

        # Get the track ID where the connecting node is located
        con_track      = GetTrackwithNode(closest_idx, Tracks_)
        if (con_track == -1):
            # print("Error could not find track, continuing,..., ", closest_idx)
            continue

        # Check if that index has changed, use that track
        con_track_dict = GetTrackDictwithNode(closest_idx, Tracks_)

        # The current node should not have more than 2 connections as its an end
        # The connecting node should not have more than 3 connections
        if (connection_count_[closest_idx] >= 3 or connection_count_[end_conn_node] >= 2):

            # Remove the old tracks from the array
            for j, t in enumerate(Tracks_):
                
                # Change the color to black
                if (t["id"] == curr_track):
                    Tracks_[j]["c"] = "black"
            # print("node already has three connecitons,skipping...")
            continue

        # Check if the current track is connected to the proposed track already
        if (set(Track["nodes"]) & set(con_track_dict["nodes"])):
            
            # Remove the old tracks from the array
            for j, t in enumerate(Tracks_):
                
                # Change the color to black
                if (t["id"] == curr_track):
                    Tracks_[j]["c"] = "black"
                    
            # print("the trying to connect both ends of track to the same track")
            continue

        # If we are proposing to connect tracks in different groups, then skip
        if (not CheckSameGroup(data, end_conn_node, closest_idx)):
            continue

        # if node-node then merge nodes and update track in Tracks
        if (closest_idx == con_track_dict["start"] or closest_idx == con_track_dict["end"]):
            # print(curr_track, con_track, closest_idx, con_track_dict["start"], con_track_dict["end"])
            
            if (con_point == "start"):
                curr_track_path.insert(0,closest_idx)
                newpath = join_tracks(curr_track_path, con_track_dict["nodes"])
            else:
                curr_track_path.append(closest_idx)
                newpath = join_tracks(curr_track_path, con_track_dict["nodes"])
        
            UpdateAndMergeTrack(curr_track, con_track, newpath, Tracks_, data)
            connected_nodes_, connections_, connection_count_ = UpdateConnections(closest_idx, end_conn_node, connected_nodes_, connections_, connection_count_)
            return False, Tracks_, connected_nodes_, connections_, connection_count_

        # Check if the proposed connection will form a cycle
        cycle  = Testcycle(end_conn_node, closest_idx ,connected_nodes_, connections_, connection_count_)

        if not cycle:

            if (con_point =="start"):
                curr_track_path.insert(0,closest_idx)
            else:
                curr_track_path.append(closest_idx)

            Track["nodes"] = curr_track_path
            connected_nodes_, connections_, connection_count_ = UpdateConnections(closest_idx, end_conn_node, connected_nodes_, connections_, connection_count_)
        else:
            continue

        # Get all connected tracks to the closest index and current track, keep track ids
        # Get node paths from all the start and end positions of the tracks find the largest one
        # Walk along 

        # Get the length either side of track
        seg1_path = GetNodePath(connected_nodes_, closest_idx, connected_nodes_[closest_idx][0])
        seg2_path = GetNodePath(connected_nodes_, closest_idx, connected_nodes_[closest_idx][1])

        # Now get the lengths and energies of the track segments
        total_length_seg1, total_energy_seg1 = GetTrackLengthEnergy(seg1_path, data)
        total_length_seg2, total_energy_seg2 = GetTrackLengthEnergy(seg2_path, data) 
        total_length_seg3, total_energy_seg3 = GetTrackLengthEnergy(curr_track_path, data) 

        # Find the delta and the primary track and add them to the new track list
        if (total_length_seg1 < total_length_seg2 and total_length_seg1 < total_length_seg3):
            AddConnectedTracks(curr_track, con_track, seg1_path, seg2_path, curr_track_path, Tracks_, data)
            return False, Tracks_, connected_nodes_, connections_, connection_count_
        
        elif ((total_length_seg2 < total_length_seg1 and total_length_seg2 < total_length_seg3)):
            AddConnectedTracks(curr_track, con_track, seg2_path, seg1_path, curr_track_path, Tracks_, data)
            return False, Tracks_, connected_nodes_, connections_, connection_count_
        
        else:

            for j, t in enumerate(Tracks_):
                
                # Change the color to black
                if (t["id"] == curr_track):
                    Tracks_[j]["c"] = "black"

            continue

    return True, Tracks_, connected_nodes_, connections_, connection_count_

# ---------------------------------------------------------------------------------------------------
# Function to walk along a track segment till we get to an end
# Accounts for forks in the track
# For this function to work, we must begin at an end e.g. 1 node connected
# this is so we can walk along and get the longest track from this point
def GetNodePath(graph_, start_node, forward_node):
    
    # Copy the dictionary of the connections
    graph = copy.deepcopy(graph_)
    
    # start the path at the first node
    path = [start_node]

    query     = forward_node # This is the node the next node in the track
    prev_node = start_node # set the start node (so first step doesnt go backwards)

    while(True):

        if (query in path):
            print("Error the path has looped, breaking,...")
            return path

        # Add the new node to the total path
        path.append(query)
        
        # Get the connected nodes to the current node
        con_nodes = graph[query]

        # We hit a end-point and it didnt loop
        if (len(con_nodes) == 1):
            return path
        
        # The node has three connections, so need to walk along each segment
        # give back the longest segment path
        if (len(con_nodes) == 3 ):
            con_nodes.remove(prev_node)
            len1 = len(GetNodePath(graph, query, con_nodes[0]))
            len2 = len(GetNodePath(graph, query, con_nodes[1]))

            if (len1 > len2):
                prev_node = query
                query = con_nodes[0]
            else:
                prev_node = query
                query = con_nodes[1]

            continue

        if (len(con_nodes) > 3 ):
            print("Error too many nodes in pathing that I was anticipating...")
            return path

        # Get the node that went in the query before so we can continue walking along
        if con_nodes[1] == prev_node:
            prev_node = query
            query = con_nodes[0]
        else:
            prev_node = query
            query = con_nodes[1]

        # Keep going,...

    print("Well, we shouldnt get here now, should we,...")
    return path

# ---------------------------------------------------------------------------------------------------
# Function to calculate the angle between two vectors
def angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Ensure the value is within [-1, 1] due to floating-point precision
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)

# ---------------------------------------------------------------------------------------------------
def CategorizeTracks(Tracks_):

    temp_length = 0
    primary_track_id = -1
    primary_nodes = []

    # Get the primary track 
    for t in Tracks_:

        if (len(t["nodes"]) > temp_length):
            temp_length = len(t["nodes"]) 
            primary_track_id = t["id"]
            primary_nodes = t["nodes"]

    # Now find tracks connected to the primary
    # if connected, label with dark-red, else give it a different colour

    for t in Tracks_:

        if (t["id"] == primary_track_id):
            t["label"] = "primary"
            t["c"] = "Teal"
            continue
        
        # Color for Brems
        if (len(t["nodes"]) == 1):
            t["c"] = "Orange"
            t["label"] = "track"
            continue

        # Check for common elements to the primary track
        # if true then it is a delta
        common_elements = set(t["nodes"]) & set(primary_nodes)

        if common_elements:
            t["c"] = "DarkRed"
            t["label"] = "track"
        else:
            t["c"] = next(color_cycle)
            t["label"] = "track"

    return Tracks_

# ---------------------------------------------------------------------------------------------------
def GetAnglesDF(df, all_visited, primary, trkid):

    # Take the mean every ten rows
    # df = Primary_Track.groupby(np.arange(len(Primary_Track)) // 10).mean()

    df['id'] = df.index
    df['primary'] = primary
    df["trkID"] = trkid
    df["event_id"] = df.event_id.iloc[0]

    # Get the diff between each row
    distances = [0]
    for i in range(1, len(df)):
        
        prev_point = df.iloc[i - 1][['x', 'y', 'z']].to_numpy()
        curr_point = df.iloc[i][['x', 'y', 'z']].to_numpy()
        distances.append(euclidean_distance(curr_point,prev_point))


    cum_distance = []
    sum_ = 0
    for d in distances:
        sum_ = sum_+d
        cum_distance.append(sum_)

    cum_distance = np.array(cum_distance)


    angles = [0]  # First point has no preceding point for angle calculation

    # Iterate through the points
    for i in range(1, len(df)):
        # Current and previous points
        prev_point = df.iloc[i - 1][['x', 'y', 'z']].to_numpy()
        curr_point = df.iloc[i][['x', 'y', 'z']].to_numpy()
        
        
        # Calculate angle between the vectors
        if i > 0:  # Skip the first vector, as there's no previous vector
            prev_vector = prev_point - df.iloc[i - 2][['x', 'y', 'z']].to_numpy()
            curr_vector = curr_point - prev_point

            angle = angle_between_vectors(prev_vector, curr_vector)
            angles.append(angle)
        else:
            angles.append(0)  # No angle for the first vector

    # Add the cumulative distances and angles as new columns
    df['cumulative_distance'] = cum_distance
    df['angle'] = angles

    # Remove nodes that have already been counted
    df = df[~df['id'].isin(all_visited)]

    return df

# ---------------------------------------------------------------------------------------------------
def GetMinima(index, all_visited_, input_data, temp_dist_matrix, R):


    distances_from_index = temp_dist_matrix[index] # distances for node to others
    sorted_indices = np.argsort(distances_from_index) # indexes sorted by smallest distance

    closest_nodes = sorted_indices[distances_from_index[sorted_indices] < R]
    
    closest_nodes = list(set(closest_nodes) - set(all_visited_))

    selected_rows = input_data.iloc[closest_nodes] # Df containing the nodes within distance

    # Compute the mean of x, y, and z columns
    mean_x = selected_rows['x'].median()
    mean_y = selected_rows['y'].median()
    mean_z = selected_rows['z'].median()
    energy_sum = selected_rows['energy'].sum()
    group_id   = int(selected_rows["group_id"].iloc[0])
    mean_point = np.array([mean_x, mean_y, mean_z, energy_sum, group_id])

    all_visited = all_visited_ + list(closest_nodes)

    return mean_point, all_visited

# ---------------------------------------------------------------------------------------------------
# Function to apply an energy threshold then redistibute the removed energy proportionally based
# on the fraction of the energy each hit has of the total remaining energy
def CutandRedistibuteEnergy(data, energy_threshold):
    # Filter out rows where energy <= energy_threshold
    removed_energy = data[data.energy <= energy_threshold]["energy"].sum()
    cut_data = data[data.energy > energy_threshold]

    # Redistribute removed energy proportionally
    cut_data["energy"] += cut_data["energy"] / cut_data["energy"].sum() * removed_energy

    # Reset the index otherwise this messes up the tracking algorithm,...
    cut_data = cut_data.reset_index(drop=True)

    return cut_data

# ---------------------------------------------------------------------------------------------------
# Function to plot connections 2D
def plot_tracks(ax, x, y, connection_count, x_label, y_label, Tracks_):
    # Filter data for markers with count 1, 0, or 3
    filtered_indices = [i for i, count in enumerate(connection_count) if count in [0, 1, 3]]
    filtered_x = [x[i] for i in filtered_indices]
    filtered_y = [y[i] for i in filtered_indices]

    # Define colors for filtered data
    colors = ["r" if connection_count[i] == 1 else 
              "orange" if connection_count[i] == 0 else 
              "darkgreen" for i in filtered_indices]

    # Scatter plot of selected points
    ax.scatter(filtered_x, filtered_y, c=colors, marker='o')

    # Plot connections
    for Track in Tracks_:
        for i in range(len(Track["nodes"]) - 1):
            start_node, end_node = Track["nodes"][i], Track["nodes"][i + 1]
            ax.plot([x[start_node], x[end_node]], [y[start_node], y[end_node]], 
                    color=Track["c"], linestyle="-")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{x_label}-{y_label} Projection')
# ---------------------------------------------------------------------------------------------------
# Function to plot 3D tracks
def plot_tracks_3D(ax, x, y, z, connection_count, Tracks_):
    # ax.set_facecolor("black")

    # Filter data for markers with count 1, 0, or 3
    filtered_indices = [i for i, count in enumerate(connection_count) if count in [0, 1, 3]]
    filtered_x = [x[i] for i in filtered_indices]
    filtered_y = [y[i] for i in filtered_indices]
    filtered_z = [z[i] for i in filtered_indices]

    # Define colors for filtered data
    colors = ["r" if connection_count[i] == 1 else 
              "orange" if connection_count[i] == 0 else 
              "darkgreen" for i in filtered_indices]

    # Scatter plot of selected points
    ax.scatter(filtered_x, filtered_y, filtered_z, c=colors, marker='o')

    # Plot connections
    for Track in Tracks_:
        for i in range(len(Track["nodes"]) - 1):
            start_node, end_node = Track["nodes"][i], Track["nodes"][i + 1]
            ax.plot([x[start_node], x[end_node]], 
                    [y[start_node], y[end_node]], 
                    [z[start_node], z[end_node]], 
                    color=Track["c"], linestyle="-")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Track Projection")

# ---------------------------------------------------------------------------------------------------
# Function to make track objects from connected nodes in the previous stages
# It creates the primary track e.g. the primary electron. Then makes delta rays on top of that
# sometimes there can be deltas on the deltas and this is taken care of. 
def MakeTracks(connection_count_, connected_nodes_, data_nodes, remaining_nodes, data, iteration, trk_ids, RebuiltTrack_):

    Track_arrays = []
    prim_track_id = -1
    prim_len = 0
    prim_track_arr = []
    prim_energy = 0

    # Get all nodes with single connections
    # These indicate the beginning of the track
    end_points = np.where(connection_count_ == 1)[0]

    # Here we remove the nodes from the list we have already considered
    end_points = [x for x in end_points if x in remaining_nodes]

    # In iteration zero, we should be considering the primary track
    # else its treated as a brem
    if (iteration == 0):
        primary_label = "Primary"
        delta_label = "Delta"
        color = "Teal"
    else:
        primary_label = "Brem"
        delta_label = "BremDelta"
        color = next(color_cycle)

    # Loop over the end points and get the track path
    for index, end_point in enumerate(end_points):
        trkpath = GetNodePath(connected_nodes_, end_point, connected_nodes_[end_point][0])
        Track_arrays.append(trkpath)

        trk_length = GetTrackLength(trkpath, data)

        # When we get the longest track this info gets overwritten
        if (trk_length > prim_len):
            prim_len = trk_length
            prim_track_id = index
            prim_track_arr = trkpath
            prim_energy = GetTrackEnergy(trkpath, data, False)
    
    # Now we are ready to create the primary track
    RebuiltTrack_.append({"id":trk_ids, "start":prim_track_arr[0], "end":prim_track_arr[-1], "length":prim_len, "energy":prim_energy, "label":primary_label, "c":color, "nodes":prim_track_arr})
    trk_ids = trk_ids + 1

    # Get all nodes with three connections in the primary track
    # These are the delta rays connected to the main track
    multi_connections = np.where(connection_count_ == 3)[0]
    prim_track_multi_connections = [x for x in multi_connections if x in prim_track_arr]

    for node in prim_track_multi_connections:
        # Need to remember what this does
        delta_node = [x for x in connected_nodes_[node] if x not in prim_track_arr] 

        if (len(delta_node) == 0):
            print("Error no delta node after filtering")
            continue

        if (len(delta_node) > 1):
            print("Error the delta node has more than one node after cut")

        delta_paths = GetDeltaPath(connected_nodes_, node , delta_node[0], 0)
    
        for t in range(len(delta_paths)):
            trkpath = delta_paths[t]
            trk_energy = GetTrackEnergy(trkpath, data, True)
            trk_length = GetTrackLength(trkpath, data)
            RebuiltTrack_.append({"id":trk_ids, "start":trkpath[0], "end":trkpath[-1], "length":trk_length, "energy":trk_energy, "label":f"{delta_label}{t}", "c":"DarkRed", "nodes":trkpath})
            trk_ids = trk_ids + 1


    # This is for single nodes
    single_points = np.where(connection_count_ == 0)[0]
    single_points = [x for x in single_points if x in remaining_nodes] # Removes ones that have already been added

    for index, single_point in enumerate(single_points):
        trkpath = [single_point]
        energy = GetTrackEnergy(trkpath, data, False)
        RebuiltTrack_.append({"id":trk_ids, "start":trkpath[0], "end":trkpath[-1], "length":0, "energy":energy, "label":"Brem", "c":"Orange", "nodes":trkpath})
        trk_ids = trk_ids + 1


    track_nodes = []
    for t in RebuiltTrack_:
        track_nodes = track_nodes + t["nodes"]
    remaining_nodes = list(set(data_nodes) - set(track_nodes))

    return RebuiltTrack_, remaining_nodes, trk_ids

# ---------------------------------------------------------------------------------------------------
def RebuildTracks(connected_nodes_, connection_count_, data):

    RebuiltTrack_ = []
    Track_arrays = []
    Accounted_nodes = []
    track_nodes = []

    data_nodes = data.index.values.tolist()
    remaining_nodes = data_nodes
    trk_ids = 0
    i = 0
    
    # Loop over and build tracks
    while remaining_nodes:
        RebuildTracks_, remaining_nodes, trk_ids = MakeTracks(connection_count_, connected_nodes_, data_nodes, remaining_nodes, data, i, trk_ids, RebuiltTrack_)
        i = i + 1

    # Quality control
    track_nodes = []
    e_sum = 0
    for t in RebuiltTrack_:
        e_sum = e_sum+t["energy"]
        track_nodes = track_nodes + t["nodes"]


    ratio = e_sum / data.energy.sum()

    if (ratio < 0.999 or ratio > 1.0001):
        print("Ratio is off:", ratio)
        return RebuiltTrack_, False

    are_equal = set(track_nodes) == set(data_nodes)

    if (not are_equal):
        print("Missing Nodes:", set(data_nodes) - set(track_nodes))
        return RebuiltTrack_, False

    return RebuiltTrack_, True

# ---------------------------------------------------------------------------------------------------
# Function to walk along a track segment till we get to an end
def GetDeltaPath(graph_, start_node, forward_node, trkidx):
    
    graph = copy.deepcopy(graph_)
    
    paths = {trkidx : [start_node]}
    
    query = forward_node
    prev_node = start_node 

    # for index,n in enumerate(range(len(graph))):
    while(True):

        paths[trkidx].append(query)
        
        # Get the connected nodes
        con_nodes = graph[query]

        # We hit a end-point and it didnt loop
        if (len(con_nodes) == 1):
            return paths
        
        if (len(con_nodes) == 3 ):
            con_nodes.remove(prev_node)

            path1 = GetDeltaPath(graph, query, con_nodes[0], 0)
            path2 = GetDeltaPath(graph, query, con_nodes[1], 0)

            len1 = len(path1[0])
            len2 = len(path2[0])

            if (len1 > len2):
                prev_node = query
                query = con_nodes[0]
                paths[trkidx+1] = path2[0]
            else:
                prev_node = query
                query = con_nodes[1]
                paths[trkidx+1] = path1[0]
            
            trkidx = trkidx+1

            continue

        if (len(con_nodes) > 3 ):
            print("Error too many nodes in pathing that I was anticipating...")

        # Get the node that went in the query before
        if con_nodes[1] == prev_node:
            prev_node = query
            query = con_nodes[0]
        else:
            prev_node = query
            query = con_nodes[1]

    print("We should not be here,...")
    return paths

# ---------------------------------------------------------------------------------------------------
# Get the length and energy of a track
def GetTrackLength(path, data):
    total_length = 0

    # Return the hit if there is only one node
    if len(path) == 0:
        return 0

    for t in range(len(path) - 1):
        point1 = data.iloc[path[t]]
        point2 = data.iloc[path[t + 1]]
        
        distance = calculate_distance(point1, point2)
        total_length += distance

    return round(total_length, 3)

# ---------------------------------------------------------------------------------------------------
# Get the length and energy of a track
def GetTrackEnergy(path, data, daughter):
    
    total_energy = 0

    if (daughter):
        path = path[1:] # Remove the first node which will be duplicated

    # Return the hit if there is only one node
    if len(path) == 0:
        return data.iloc[path[0]]['energy']

    for t in range(len(path)):
        point = data.iloc[path[t]]
        total_energy += point['energy']
    
    return total_energy

# ---------------------------------------------------------------------------------------------------
# Get the median distances between nodes:
def GetMedianNodeDistances(df):

    # Extract x, y, z coordinates as a numpy array
    points = df[['x', 'y', 'z']].values

    # Calculate the Euclidean distance between each pair of points
    distances = cdist(points, points, metric='euclidean')

    # Replace the diagonal (self-distance) with np.inf so it doesn't get selected as minimum
    np.fill_diagonal(distances, np.inf)

    # Find the minimum distance for each row
    min_distances = distances.min(axis=1)

    # Calculate the mean of the minimum distances
    median_distance = np.median(min_distances)

    print("Median distance to the closest row:", median_distance)

    if (median_distance <= 0):
        print("Median distance was zero, so setting to default value of 15") 
        median_distance = 15
    elif (median_distance ==np.inf):
        print("Median distance was infinate, so setting to default value of 15")
        median_distance = 15
    
    return median_distance

# ---------------------------------------------------------------------------------------------------
# Calculate the angular variables averaged along track
# Tortuosity: defined as the total path length along actual track
# divided by the straight line distance
# Squiglicity: defined as the sum of the normal distances to the best
# straight line fit to the nodes divided by the straight line distance
def CalcAngularVars(df_angles, Tortuosity_dist):

    # df_angles['distance_diff'] = df_angles.groupby(['event_id', 'trkID'])['cumulative_distance'].diff().fillna(0)

    df_angles["Tortuosity"] = 1.0
    Tortuosity = []
    df_angles["Squiglicity"] = 1.0
    Squiglicity = []
    # df_angles["RMS"] = 0.0
    # RMS = []

    for trkID in df_angles.trkID.unique():

        # Get the track -- should be fine to reset the index as long as the ordering is preserved to df_angles
        trk_df = df_angles[df_angles.trkID == trkID].reset_index(drop=True) 

        # For the deltas and brems, use 10% of the track to compute quanitites
        if (trk_df.primary.iloc[0] != 1):
            Tortuosity_dist = max(trk_df.cumulative_distance/10)
            if (Tortuosity_dist == 0):
                Tortuosity_dist = 10


        # Loop over the nodes in the track
        for index in range(len(trk_df)):

            # Start with the current index
            valid_rows = [index]  
            cumulative_distance = 0.0  

            # Iterate forward and compute cumulative distance
            for i in range(index + 1, len(trk_df)):
                prev_row = trk_df.iloc[valid_rows[-1]]
                curr_row = trk_df.iloc[i]

                step_distance = ((curr_row.x - prev_row.x) ** 2 +
                                (curr_row.y - prev_row.y) ** 2 +
                                (curr_row.z - prev_row.z) ** 2) ** 0.5

                if cumulative_distance + step_distance > Tortuosity_dist:
                    break  # Stop once threshold is exceeded

                cumulative_distance += step_distance
                valid_rows.append(i) # add the rows to calculate tortuosity

            # Reset for backward direction
            cumulative_distance = 0.0  

            # Iterate backward
            for i in range(index - 1, -1, -1):  # Iterate to the first row (index 0)
                prev_row = trk_df.iloc[valid_rows[0]]  # First row in valid list
                curr_row = trk_df.iloc[i]

                step_distance = ((curr_row.x - prev_row.x) ** 2 +
                                (curr_row.y - prev_row.y) ** 2 +
                                (curr_row.z - prev_row.z) ** 2) ** 0.5

                if cumulative_distance + step_distance > Tortuosity_dist:
                    break

                cumulative_distance += step_distance
                valid_rows.insert(0, i)  # Add at the beginning to maintain order

            # Select only the valid rows
            temp_df = trk_df.loc[valid_rows]

            point1 = temp_df.iloc[0]
            point2 = temp_df.iloc[-1]
            segment_length = calculate_distance(point1, point2)

            # Avoids division by zero
            if (segment_length == 0):
                segment_length = 1

            # Tortuosity
            # Get the diff between each row
            cum_distance = 0
            for i in range(1, len(temp_df)):
                
                prev_point = temp_df.iloc[i - 1][['x', 'y', 'z']].to_numpy()
                curr_point = temp_df.iloc[i][['x', 'y', 'z']].to_numpy()
                cum_distance+=euclidean_distance(curr_point,prev_point)

            temp_tortuoisty = cum_distance/segment_length
            if (temp_tortuoisty) == 0:
                temp_tortuoisty = 1.0
            Tortuosity.append(temp_tortuoisty)


            # Squiglicity
            points = temp_df[['x', 'y', 'z']].values # Get the points
            
            # Step 1: Compute centroid (a point on the best-fit line)
            A = np.mean(points, axis=0)
            
            # Step 2: Compute best-fit direction vector using SVD
            U, S, Vt = np.linalg.svd(points - A)
            D = Vt[0]  # First right-singular vector (best-fit direction)

            temp_df['distance_to_line'] = [point_to_line_distance(P, A, D) for P in points]

            # Step 4: Sum of distances
            cum_distance = temp_df['distance_to_line'].sum()

            Squiglicity.append(cum_distance/segment_length)

            # RMS
            # temp_df['distance_to_line2']= temp_df['distance_to_line']*temp_df['distance_to_line']
            # RMS.append(np.sqrt(temp_df['distance_to_line2'].mean()))

    df_angles["Tortuosity"]  = Tortuosity
    df_angles["Squiglicity"] = Squiglicity
    # df_angles["RMS"]         = RMS


    return df_angles

# ---------------------------------------------------------------------------------------------------
# Function to calculate perpendicular distances from SVD decomp of a set of points
# used in Squiglicity Calculation
def point_to_line_distance(P, A, D):
    return np.abs(np.linalg.norm(np.cross(P - A, D)) / np.linalg.norm(D))
# ---------------------------------------------------------------------------------------------------
# Function to return the blob with the biggest energy
# will swap if blob1 and blob2 are the otherway round
# blob1: blob1 energy
# blob2: blob2 energy
def SortEnergy(blob1, blob2):

    # Extra flag to indicate the ends were swapped
    if (blob1 > blob2):
        return blob1, blob2, False
    else:
        return blob2, blob1, True

# ---------------------------------------------------------------------------------------------------
# Function calculates overall track properties such as the blob energies and end tortuosities
# df                 : dataframe containing angles
# trkID              : the id of the track 
# primary            : (bool 1 or 0) to indicate if the track is the primary
# p_start            : the start id node
# p_end              : the end id node
# eventid            : the event id
# distance_threshold : the distance to calculate the blob energy
# radius_threshold   : the radius of sphere to calculate blob energy
# T_threshold        : the distance along the track to calculate the tortuosity
# pressure           : (int) the gas pressure
def GetTrackProperties(df, trkID, primary, p_start, p_end, eventid, distance_threshold, radius_threshold, T_threshold, pressure):

    # Now Get the energy of the primary end points radius method
    blob1R   = GetBlobEnergyRadius(df, p_start, radius_threshold)
    blob2R   = GetBlobEnergyRadius(df, p_end,   radius_threshold)
    
    # Now Get the energy of the primary end points length method
    blob1, blob2 = GetBlobEnergyLength(df, distance_threshold) # Uses cumulative distance
    
    # Get the tortuosity
    T1, T2   = GetEndVariableMean(df, T_threshold, "Tortuosity")

    # Get the squiglicity
    S1, S2   = GetEndVariableMean(df, T_threshold, "Squiglicity")
    
    # Create a new DataFrame to append
    properties_df = pd.DataFrame({
        "event_id"       : [eventid],
        "trkID"          : [trkID],
        "primary"        : [primary],
        "start"          : [p_start],
        "end"            : [p_end],
        "blob1"          : [blob1],
        "blob2"          : [blob2],
        "blob1R"         : [blob1R],
        "blob2R"         : [blob2R],
        "Tortuosity1"    : [T1], 
        "Tortuosity2"    : [T2],
        "Squiglicity1"   : [S1],
        "Squiglicity2"   : [S2]
    })

    properties_df["trkID"] = properties_df["trkID"].astype(int)

    return properties_df

# ---------------------------------------------------------------------------------------------------
# Function that gets the area of the tortuosity near the ends of the track
# we multiply by pressure to make the value simular for what it is at 1bar
# df          : the dataframe containing the tortuosity and cumulative distance
# T_threshold : the cumulative distance threshold at the ends to calculate from
# pressure           : (int) the gas pressure
def GetEndVariableArea(df, T_threshold, pressure, var_name):

    if len(df) == 1:
        return 0, 0

    df_var1 = df[df.cumulative_distance < T_threshold]
    
    # Extend if there was only 1 or zero rows 
    if len(df_var1) <= 1:
        df_var1 = df.head(2)

    var1 = np.trapz(df_var1[f"{var_name}"], df_var1["cumulative_distance"]*pressure)

    end_threshold = max(df.cumulative_distance) - T_threshold
    df_var2 = df[df['cumulative_distance'] > end_threshold]
    
    # Extend if there was only 1 or zero rows 
    if len(df_var2) <= 1:
        df_var2 = df.tail(2)

    # print("end_tresh:", end_threshold, max(df.cumulative_distance), T_threshold, pressure, len(df_var2))
    var2 = np.trapz(df_var2[f"{var_name}"], (df_var2["cumulative_distance"] - end_threshold)*pressure ) # get the area

    if var1 < 0:
        print(f"{var_name}1 was less than zero, setting to 0")
        var1 = 0.0
    if var2 < 0:
        print(f"{var_name}2 was less than zero, setting to 0")
        var2 = 0.0

    return var1, var2
# ---------------------------------------------------------------------------------------------------
# Function that gets the mean of the tortuosity near the ends of the track
# df          : the dataframe containing the tortuosity and cumulative distance
# T_threshold : the cumulative distance threshold at the ends to calculate from
def GetEndVariableMean(df, T_threshold, var_name):
    
    if len(df) == 1:
        return 0, 0

    df_var1 = df[df.cumulative_distance < T_threshold]
    
    # Extend if there was only 1 or zero rows 
    if len(df_var1) <= 1:
        df_var1 = df.head(2)
    
    var1 = df_var1[f"{var_name}"].mean()

    end_threshold = max(df.cumulative_distance) - T_threshold
    df_var2 = df[df['cumulative_distance'] > end_threshold]

    # Extend if there was only 1 or zero rows 
    if len(df_var2) <= 1:
        df_var2 = df.tail(2)

    var2 = df_var2[f"{var_name}"].mean()

    if var1 < 0:
        print(f"{var_name}1 was less than zero, setting to 0")
        var1 = 0.0
    if var2 < 0:
        print(f"{var_name}2 was less than zero, setting to 0")
        var2 = 0.0

    return var1, var2

# ---------------------------------------------------------------------------------------------------
# Function gets the energy based on a sphere of radius radius_threshold
def GetBlobEnergyRadius(df, p_start, radius_threshold):
    # Get coordinates where id is p_start
    start_coord = df[df['id'] == p_start][['x', 'y', 'z']].values

    # Calculate the Euclidean distance from each row to each row with id == p_start
    distances = np.sqrt(((df[['x', 'y', 'z']].values[:, None] - start_coord) ** 2).sum(axis=2))

    # Find rows where any distance to id == p_start rows is less than the threshold
    mask = (distances < radius_threshold).any(axis=1)
    result = df[mask]
    return result.energy.sum()

# ---------------------------------------------------------------------------------------------------
# Gets the blob energy walking along the track length of distance distance_threshold
def GetBlobEnergyLength(df, distance_threshold):
    df_E1 = df[df.cumulative_distance < distance_threshold]
    E1 = df_E1["energy"].sum()

    end_threshold = max(df.cumulative_distance) - distance_threshold
    df_E2 = df[df['cumulative_distance'] > end_threshold]
    E2 = df_E2["energy"].sum()
    return E1, E2

# ---------------------------------------------------------------------------------------------------
# Function call to get the track metadata
# df_angles          : the dataframe containing track angles
# RebuiltTrack       : the dict containing overall track infomation
# distance_threshold : the distance to calculate the blob energy
# radius_threshold   : the radius of sphere to calculate blob energy
# T_threshold        : the distance along the track to calculate the tortuosity
# pressure           : (int) the gas pressure
def GetTrackdf(df_angles, RebuiltTrack, distance_threshold, radius_threshold, T_threshold, pressure):
    Track_df = []

    # print("distance_threshold, radius_threshold, T_threshold, pressure: ", distance_threshold, radius_threshold, T_threshold, pressure)
    
    # loop over the tracks
    for t in RebuiltTrack:

        p_start = t["start"]
        p_end   = t["end"]

        # Select only specific variables to store for the track properties
        filtered_data = {key: t[key] for key in ['id', 'length', 'energy', 'label']}

        eventid = df_angles.event_id.iloc[0]
        primary_id = df_angles[df_angles.trkID == t["id"]]["primary"].iloc[0]

        properties_df = GetTrackProperties(df_angles[df_angles.trkID == t["id"]], t["id"], primary_id, p_start, p_end, eventid, distance_threshold, radius_threshold, T_threshold, pressure)

        # Convert to DataFrame
        df = pd.DataFrame([filtered_data])
        df.rename(columns={'id': 'trkID'}, inplace=True)
        df = properties_df.merge(df, on='trkID', how='inner')
        df = df[["event_id", "trkID","primary", "start", "end", "length", "energy", "blob1", "blob2", "blob1R", "blob2R", "Tortuosity1", "Tortuosity2", "Squiglicity1","Squiglicity2", "label"]]

        Track_df.append(df)

    Track_df = pd.concat(Track_df)
    return Track_df

# ---------------------------------------------------------------------------------------------------
# If a delta/brem has an position to close to the blob ends, then combine that info into energy and tortuosity.
# we add the total energy of the secondary
def UpdateTrackMeta(Track_df, df_angles, distance):

    df = Track_df.copy()

    prim_df = df[df.primary == 1]

    blob1_energy  = [prim_df.blob1.iloc[0]]
    blob2_energy  = [prim_df.blob2.iloc[0]]
    blob1R_energy = [prim_df.blob1R.iloc[0]]
    blob2R_energy = [prim_df.blob2R.iloc[0]]
    Tortuosity1   = [prim_df.Tortuosity1.iloc[0]]
    Tortuosity2   = [prim_df.Tortuosity2.iloc[0]]
    Squiglicity1  = [prim_df.Squiglicity1.iloc[0]]
    Squiglicity2  = [prim_df.Squiglicity2.iloc[0]]
    
    prim_start = df_angles[df_angles['id'] == prim_df.start.iloc[0]][['x', 'y', 'z']].values
    prim_end   = df_angles[df_angles['id'] == prim_df.end.iloc[0]][['x', 'y', 'z']].values

    # Loop over the secondary tracks
    for t in df[df.primary == 0].trkID.unique():

        # The secondary track
        trk_df = df[df.trkID == t]

        trk_start = df_angles[df_angles['id'] == trk_df.start.iloc[0]][['x', 'y', 'z']].values
        trk_end   = df_angles[df_angles['id'] == trk_df.end.iloc[0]][['x', 'y', 'z']].values

        # Check delta/brem to the blob1 pos
        dist_blob1 = euclidean_distance(prim_start, trk_start)
        if (dist_blob1 < distance):
            print(f"Adding {trk_df.label.iloc[0]} energy to blob1 as dist was {dist_blob1}")
            blob1_energy.append(trk_df.energy.iloc[0])
            blob1R_energy.append(trk_df.energy.iloc[0])
            Tortuosity1.append(trk_df.Tortuosity1.iloc[0])
            Squiglicity1.append(trk_df.Squiglicity1.iloc[0])
            continue

        # Check delta/brem to the blob2 pos
        dist_blob2   = euclidean_distance(prim_end, trk_start)
        if (dist_blob2 < distance):
            print(f"Adding trk {t} {trk_df.label.iloc[0]} energy to blob2 as dist was {dist_blob2}")
            blob2_energy.append(trk_df.energy.iloc[0])
            blob2R_energy.append(trk_df.energy.iloc[0])
            Tortuosity2.append(trk_df.Tortuosity2.iloc[0])
            Squiglicity2.append(trk_df.Squiglicity2.iloc[0])
            continue

    # remove any nan from the variables e.g. if there was bad delta information
    blob1_energy   = [x for x in blob1_energy   if not np.isnan(x)]
    blob2_energy   = [x for x in blob2_energy   if not np.isnan(x)]
    blob1R_energy  = [x for x in blob1R_energy  if not np.isnan(x)]
    blob2R_energy  = [x for x in blob2R_energy  if not np.isnan(x)]
    Tortuosity1    = [x for x in Tortuosity1    if not np.isnan(x)]
    Tortuosity2    = [x for x in Tortuosity2    if not np.isnan(x)]
    Squiglicity1   = [x for x in Squiglicity1   if not np.isnan(x)]
    Squiglicity2   = [x for x in Squiglicity2   if not np.isnan(x)]

    # Here we need to add an additional check to see which blob energy was greater and then swap the columns accordingly
    blob1_energy_sum = np.float32(sum(blob1_energy))
    blob2_energy_sum = np.float32(sum(blob2_energy))
    
    # See if we swap
    blob1_energy_sum, blob2_energy_sum, swapped_flag = SortEnergy(blob1_energy_sum, blob2_energy_sum)

    if (not swapped_flag):
        df.loc[df['primary'] == 1, 'blob1']  = blob1_energy_sum
        df.loc[df['primary'] == 1, 'blob2']  = blob2_energy_sum
        df.loc[df['primary'] == 1, 'blob1R'] = np.float32(sum(blob1R_energy))
        df.loc[df['primary'] == 1, 'blob2R'] = np.float32(sum(blob2R_energy))
        df.loc[df['primary'] == 1, 'Tortuosity1'] = sum(Tortuosity1)
        df.loc[df['primary'] == 1, 'Tortuosity2'] = sum(Tortuosity2)
        df.loc[df['primary'] == 1, 'Squiglicity1'] = sum(Squiglicity1)
        df.loc[df['primary'] == 1, 'Squiglicity2'] = sum(Squiglicity2)
    else:
        print("Swapping the blob names")
        df.loc[df['primary'] == 1, 'blob1']  = blob1_energy_sum # these have already been swapped so keep
        df.loc[df['primary'] == 1, 'blob2']  = blob2_energy_sum
        df.loc[df['primary'] == 1, 'blob1R'] = np.float32(sum(blob2R_energy)) # swap the blob E radius method
        df.loc[df['primary'] == 1, 'blob2R'] = np.float32(sum(blob1R_energy))
        df.loc[df['primary'] == 1, 'Tortuosity1'] = sum(Tortuosity2) # swap the tortosity
        df.loc[df['primary'] == 1, 'Tortuosity2'] = sum(Tortuosity1)
        df.loc[df['primary'] == 1, 'Squiglicity1'] = sum(Squiglicity2) # swap the squiglicity
        df.loc[df['primary'] == 1, 'Squiglicity2'] = sum(Squiglicity1)

        # Swap the ends too
        temp_start = df.loc[df['primary'] == 1, 'start']
        temp_end   = df.loc[df['primary'] == 1, 'end']
        df.loc[df['primary'] == 1, 'start'] = temp_end
        df.loc[df['primary'] == 1, 'end']   = temp_start

    return df
# ---------------------------------------------------------------------------------------------------
# Function to swap variables so that var1>var2
def SwapVariables(df, var1, var2):
        df[var1], df[var2] = np.where( df[var1] >= df[var2], (df[var1], df[var2]), (df[var2], df[var1]))
        return df
# ---------------------------------------------------------------------------------------------------
# A simpler approach to defining end-point variables
# All higher variables are organized so that blob1 is the highest and blob2 is the lowest
# Do keep energies across all separate tracks unique
def UpdateTrackMeta2(Track_df):

    df = Track_df.copy()

    df = SwapVariables(df, "blob1", "blob2")
    df = SwapVariables(df, "blob1R", "blob2R")
    df = SwapVariables(df, "Tortuosity1", "Tortuosity2")
    df = SwapVariables(df, "Squiglicity1", "Squiglicity2")

    return df

# ---------------------------------------------------------------------------------------------------
# Master function to run the tracking algorithm
# the new dataframe will contain the following new columns:
# id, primary, trkID, cumulative_distance, angle
# data     : the input dataframe containig x,y,z,energy
# cluster  : (int 1 or 0) if running over diffused files then this clusters the track
#          so that we can run over the tracking algorithm over the trunk of the track
# pressure : the pressure of the track (integer in bar). Cuts are scaled from 1 bar to this pressure
# diffusion: "nodiff", "5percent", "0.5percent", "0.25percent". "0.1percent", "0.05percent"
#            this is the diffusion amount of the tracks. This tunes the clustering amount. 
# sort flag: sometimes the tracking algorithm fails. This changes how the hits are ordered
#            which will help the algorithm converge better
def RunTracking(data, cluster, pressure, diffusion, sort_flag):

    Diff_smear, energy_threshold, diff_scale_factor, radius_sf, group_sf, Tortuosity_dist, voxel_size, det_half_length = InitializeParams(pressure, diffusion)
    print("Diffussion smear is: ",        Diff_smear,            "mm/sqrt(cm)")
    print("Energy threshold is: ",        1000*energy_threshold, "keV")
    print("diffision scale factor is: ",  diff_scale_factor)
    print("Radius scale factor is: ",     radius_sf)
    print("Hit grouping factor is: ",     group_sf)
    print("Tortuosity distance scale is:", Tortuosity_dist)
    print("The voxel size is:",           voxel_size)
    print("The det half_length is: ",     det_half_length)

    # If there are overlapping voxels, merge them. Otherwise the energy gets messed up
    data = (data.groupby(["event_id", "x", "y", "z"], as_index=False)["energy"].sum())
    # display(data)

    if ("group_id" in data.columns):
        data = data[['event_id', 'x', 'y', 'z',"energy", "group_id"]]
    else:
        data = data[['event_id', 'x', 'y', 'z',"energy"]]


    # Sort this way as input always
    data = data.sort_values(by=['x', "y", "z"]).reset_index(drop=True) 

    # Cluster the data if required
    if (cluster):
        # Also only cluster if enough points in the dataframe
        if (len(data)> 30):
            data =  RunClustering(data, pressure, diffusion)
        else:
            print("Skipping Clustering due to not enough points")
    
    # If clustering has not been applied then we assume it was nodiff sample
    # in this case apply grouping to generate the column
    if ("group_id" not in data.columns):
        print("Grouping has not been applied yet so run grouping function,...")
        mean_sigma_group = group_sf*Diff_smear*np.sqrt(0.1*data.z.mean())
        if (mean_sigma_group < 1.5*voxel_size):
            mean_sigma_group = 1.5*voxel_size
        data = GroupHits(data, mean_sigma_group)

    # then sort it based on the x,y,z
    if (sort_flag == 0):
        data = data.sort_values(by=['x', "y", "z"]).reset_index(drop=True)
    elif (sort_flag == 1):
        data = data.sort_values(by=['y', "z", "x"]).reset_index(drop=True)
    else:
        data = data.sort_values(by=['z', "x", "y"]).reset_index(drop=True)

    # Calculate the distance matrix
    dist_matrix = distance_matrix(data[['x', 'y', 'z']], data[['x', 'y', 'z']])

    # Initialize connections counter, keeps track of number of connections to each index
    connection_count = np.zeros(len(data), dtype=int)

    # This is a dict, format is
    # index : [connected node 1, connected node 2,...]
    connected_nodes = {}
    connections = []

    # Tunable parameters
    Median_dist = GetMedianNodeDistances(data) # Median distance between nodes
    init_dist_thresh = Median_dist*2 # max distance for initial connections [mm]
    incr_dist_thresh = np.linspace(1, Median_dist*radius_sf, 15, dtype=int) # Second stage, look for closest nodes, then slowly increase threshold [mm]
    incr_dist_thresh = np.unique(incr_dist_thresh)
    print("Distances to iterate over", incr_dist_thresh)

    for i in range(len(data)):
        # Find the index of the closest node (excluding itself)
        # closest_idx = np.argpartition(dist_matrix[i], 1)[1]
        closest_idx = np.argsort(dist_matrix[i])[1]
        
        # Check if the connection already exists 
        if closest_idx not in connected_nodes.get(i, []) and i not in connected_nodes.get(closest_idx, []):

            # Check the proposed node has 0 or 1 connection, the connection is within the dist threshold and is within the same group
            if (connection_count[closest_idx] <= 1 and connection_count[i] <= 1 and dist_matrix[i][closest_idx] < init_dist_thresh and CheckSameGroup(data, i, closest_idx)):
                
                cycle  = Testcycle(i, closest_idx ,connected_nodes, connections, connection_count)
                
                # Add connection between node i and closest_idx if it doesnt form a cycle
                if (not cycle):
                    connected_nodes, connections, connection_count = UpdateConnections(i, closest_idx, connected_nodes, connections, connection_count)

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

                    # Check if the index is not itelf and the connection count of the closest index is 1, also that the proposed node is in the same group
                    if closest_idx != i and connection_count[closest_idx] <= 1 and connection_count[i] <= 1 and closest_idx not in connected_nodes.get(i, []) and i not in connected_nodes.get(closest_idx, []) and CheckSameGroup(data, i, closest_idx): 
                        
                        if dist_matrix[i][closest_idx] < dist:

                            cycle  = Testcycle(i, closest_idx ,connected_nodes, connections, connection_count)
                            
                            if not cycle:
                                connected_nodes, connections, connection_count = UpdateConnections(i, closest_idx, connected_nodes, connections, connection_count)
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


    # print(GetMedianNodeDist(Tracks, data))

    dist_threshold = 4*GetMedianNodeDist(Tracks, data)
    print("Dist Thresh",dist_threshold)

    # Add in any nodes without connections to the tracks as gammas and re-label other tracks as gammas
    AddConnectionlessNodes(connection_count, Tracks, data)

    finished = False  # Initial state

    q = 0
    while not finished:
        finished, Tracks, connected_nodes, connections, connection_count = ConnectTracks(Tracks, connected_nodes, connections, connection_count, dist_matrix, dist_threshold, data)
        q=q+1


    # Redo the track building
    Tracks = []
    Tracks, pass_flag = RebuildTracks(connected_nodes, connection_count, data)
    # print(len(connected_nodes), connected_nodes)

    print("Pass Flag:",pass_flag)
   
    # return if the event did not pass
    if (not pass_flag):
        
        # The NEXT1t analysis has a central cathode to account for
        if diffusion == "next1t":
            contained = CheckHitBoundsNext1t(data)
        else:
            contained = CheckHitBounds(data, det_half_length-20, +20, det_half_length*2.0-20)
        
        return data, Tracks, connected_nodes, connection_count, pass_flag, contained

    # Function to get track topo info
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


    df_angles = CalcAngularVars(df_angles, Tortuosity_dist)  # Add the tortuosity and squiglicity

    # The NEXT1t analysis has a central cathode to account for
    if diffusion == "next1t":
        contained = CheckHitBoundsNext1t(data)
    else:
        contained = CheckHitBounds(data, det_half_length-20, +20, det_half_length*2.0-20)

    print(df_angles)
    return df_angles, Tracks, connected_nodes, connection_count, pass_flag, contained
# ---------------------------------------------------------------------------------------------------
def Cluster(input_data, R):

    node_centers = []
    all_visited = []
    indexes = input_data.index.values
    indexes_set = set(indexes)

    temp_dist_matrix = distance_matrix(input_data[['x', 'y', 'z']], input_data[['x', 'y', 'z']])

    for i in range(len(input_data)):

        all_visited_set = set(all_visited)

        # Convert arrays to sets and perform the difference
        filtered_indexes = list(indexes_set - all_visited_set)

        if not filtered_indexes:
            break

        # random_index = np.random.choice(filtered_indexes)
        random_index = filtered_indexes[0]
        median, all_visited = GetMinima(random_index, all_visited, input_data, temp_dist_matrix, R)

        node_centers.append(median)

    node_centers_df  = pd.DataFrame(node_centers, columns=['x', 'y', 'z', 'energy', "group_id"])
    node_centers_df["group_id"] = node_centers_df["group_id"].astype(int)

    return node_centers_df

# ---------------------------------------------------------------------------------------------------
def RunClustering(node_centers_df, pressure, diffusion):

    Diff_smear, energy_threshold, diff_scale_factor, radius_sf, group_sf, Tortuosity_dist, voxel_size, det_size = InitializeParams(pressure, diffusion)

    event_id = node_centers_df.event_id.iloc[0]

    # Apply energy threshold and redistribute energy
    df_merged = CutandRedistibuteEnergy(node_centers_df, energy_threshold)
    # print(df_merged)

    # define the radius to cluster by
    mean_sigma = diff_scale_factor*Diff_smear*np.sqrt(0.1*node_centers_df.z.mean())
    if (mean_sigma < 1.5*voxel_size):
        mean_sigma = 1.5*voxel_size

    # Use fixed value since voxels are same size in next1t analysis
    if (diffusion == "next1t"):
        mean_sigma=6

    print("Mean Sigma is:", mean_sigma)

    # Apply hit grouping
    mean_sigma_group = group_sf*Diff_smear*np.sqrt(0.1*node_centers_df.z.mean())
    if (mean_sigma_group < 1.5*voxel_size):
        mean_sigma_group = 1.5*voxel_size

    # Use fixed value since voxels are same size in next1t analysis
    if diffusion == "next1t":
        mean_sigma_group = 10

    df_merged = GroupHits(df_merged, mean_sigma_group)

    # Run the clustering
    node_centers_df = []

    for gid in sorted(df_merged.group_id.unique()):
        temp_df = df_merged[df_merged.group_id == gid]
        temp_df.reset_index(drop=True, inplace=True)
        node_centers_df.append(Cluster(temp_df, mean_sigma))

    node_centers_df = pd.concat(node_centers_df, ignore_index=True)
    node_centers_df["event_id"] = event_id

    # Bin the data

    # Create the bins ---- 
    xbw  = mean_sigma
    xmin = -det_size - mean_sigma/2 
    xmax = det_size  + mean_sigma/2

    ybw  = mean_sigma
    ymin = -det_size - mean_sigma/2 
    ymax = det_size  + mean_sigma/2

    # This shifts the z pos of the events so 0 is at anode
    # can set this to zero
    z_shift = det_size
    # z_shift = 0

    zbw=mean_sigma
    zmin=-det_size + z_shift - mean_sigma/2 
    zmax=det_size + z_shift + mean_sigma/2

    # bins for x, y, z
    xbins = np.arange(xmin, xmax+xbw, xbw)
    ybins = np.arange(ymin, ymax+ybw, ybw)
    zbins = np.arange(zmin, zmax+zbw, zbw)

    # center bins for x, y, z
    xbin_c = xbins[:-1] + xbw / 2
    ybin_c = ybins[:-1] + ybw / 2
    zbin_c = zbins[:-1] + zbw / 2

    databin = node_centers_df.copy()

    databin["event_id"] = event_id

    # Now lets bin the data
    databin['x_smear'] = pd.cut(x=databin['x'], bins=xbins,labels=xbin_c, include_lowest=True)
    databin['y_smear'] = pd.cut(x=databin['y'], bins=ybins,labels=ybin_c, include_lowest=True)
    databin['z_smear'] = pd.cut(x=databin['z'], bins=zbins,labels=zbin_c, include_lowest=True)

    # Drop rows with any NaN values
    databin = databin.dropna()

    # Dictionary to store results
    aggregated_data = {}

    # Iterate through the DataFrame row by row
    for _, row in databin.iterrows():
        key = (row['event_id'], row['x_smear'], row['y_smear'], row['z_smear'], row['group_id'])
        
        if key not in aggregated_data:
            # Initialize the aggregation for a new group
            aggregated_data[key] = {
                'x_sum': row['x'],
                'y_sum': row['y'],
                'z_sum': row['z'],
                'energy_sum': row['energy'],
                'group_id' : row['group_id'],
                'count': 1
            }
        else:
            # Update existing group values
            aggregated_data[key]['x_sum'] += row['x']
            aggregated_data[key]['y_sum'] += row['y']
            aggregated_data[key]['z_sum'] += row['z']
            aggregated_data[key]['energy_sum'] += row['energy']
            aggregated_data[key]['group_id'] = row['group_id']
            aggregated_data[key]['count'] += 1

    # Convert aggregated data into a DataFrame
    result = []
    for key, values in aggregated_data.items():
        event_id, x_smear, y_smear, z_smear, group_id = key
        result.append({
            'event_id': event_id,
            'x_smear': x_smear,
            'y_smear': y_smear,
            'z_smear': z_smear,
            'x': values['x_sum'] / values['count'],   # Mean x
            'y': values['y_sum'] / values['count'],   # Mean y
            'z': values['z_sum'] / values['count'],   # Mean z
            'energy': values['energy_sum'],           # Sum energy
            'group_id' : int(group_id)
        })

    # Create final DataFrame
    databin = pd.DataFrame(result)

    databin["event_id"] = databin["event_id"].astype('int')

    return databin

# ---------------------------------------------------------------------------------------------------
# Function to group hits based on proximity, returns a new row to the dataframe based on the hit group
# it uses the skikit learn DB scan tool
# df: input dataframe, returns a new column group_id with associated hits
# threshold: a maximum distance which hits can be separated from before categorizing to a new group
def GroupHits(df, threshold):

    # Convert to NumPy array for clustering
    coords = df[["x", "y", "z"]].to_numpy()

    # Apply DBSCAN
    db = DBSCAN(eps=threshold, min_samples=1).fit(coords)

    # Add group labels to the original DataFrame
    df["group_id"] = db.labels_

    if (len(df.group_id.unique()) > 8):
        print("Running grouping again new mean sigma is:", threshold*10)
        df = GroupHits(df, threshold*10)

    return df
# ---------------------------------------------------------------------------------------------------
# Function checks if the nodes are in the same group. 
# df: the input dataframe, should contain the group_id
# node1/2: the index in the dataframe to check 
def CheckSameGroup(df, node1, node2):
    group1 = df.group_id.iloc[node1]
    group2 = df.group_id.iloc[node2]

    return group1 == group2

# ---------------------------------------------------------------------------------------------------
# Function to initalize parameter cuts
# inputs: pressure and diffusion amount
# outputs: Diff_smear - The amont of diffusion that was applied for given pressure/CO2 mix
#                       this is used to calculate clustering size estimation
#         energy_threshold - this removes hits with an energy below this val and redistibutes its
#                            energy amongst the remaining hits proportationally
#         diff_scale_factor - This addtional adjustment scales the radius in clustering by this amount
#         radius_sf - The no diffusion files have a lot of nodes so skew the median node distance.
#                     factor helps to adjust that
#         group_sf -  This scales the distance used to group hit clusters together
#         Tortuosity_dist - this is the length scale to calculate tortuosity
#         voxel_size -- the size of binning used in the smear code
#         det_half_length - the half-length of the detector
def InitializeParams(pressure, diffusion):

    Diff_smear        = 0.0 # mm / sqrt(cm)
    energy_threshold  = 0   # MeV
    diff_scale_factor = 7
    radius_sf         = 7 
    group_sf          = 3
    Tortuosity_dist   = 70
    voxel_size        = 10

    # Calculate the detector half-length
    density = 5.987*pressure
    M = 1000/0.9
    det_half_length = 1000*np.cbrt((4 * M) / (np.pi * density))/2.0

    # This is acutally 10 % Helium
    if (diffusion == "0.05percent"):

        voxel_size        = 20

        if (pressure == 1 or pressure == 5):
            Diff_smear        = 2.0/np.sqrt(pressure)
            energy_threshold  = 0.0002
            diff_scale_factor = 5
            radius_sf         = 7
            group_sf          = 3
            Tortuosity_dist   = 0.05*3500/pressure
        elif (pressure == 10):
            Diff_smear        = 2.0/np.sqrt(pressure)
            energy_threshold  = 0.0002
            diff_scale_factor = 3
            radius_sf         = 7
            group_sf          = 5
            Tortuosity_dist   = 0.05*3500/pressure
        elif (pressure == 15):
            Diff_smear        = 2.0/np.sqrt(pressure)
            energy_threshold  = 0.0002
            diff_scale_factor = 3
            radius_sf         = 7
            group_sf          = 7
            Tortuosity_dist   = 0.05*3500/pressure
        else:
            Diff_smear        = 2.0/np.sqrt(pressure)
            energy_threshold  = 0.0002
            diff_scale_factor = 3
            radius_sf         = 7
            group_sf          = 7
            Tortuosity_dist   = 0.05*3500/pressure
    
    elif (diffusion == "nodiff"):
        Diff_smear        = 0.1
        energy_threshold  = 0.0
        diff_scale_factor = 7
        radius_sf         = 10
        group_sf          = 10
        Tortuosity_dist   = 0.02*3500/pressure
        voxel_size        = 5
    
    elif (diffusion == "0.1percent"):
        Diff_smear        = 1.0
        energy_threshold  = 0.0004
        diff_scale_factor = 5
        radius_sf         = 7
        group_sf          = 3
        Tortuosity_dist   = 0.05*3500/pressure
        voxel_size        = 20
    
    elif (diffusion == "0.25percent"):
        Diff_smear        = 0.703 
        energy_threshold  = 0.0004
        diff_scale_factor = 4
        radius_sf         = 7
        group_sf          = 5
        Tortuosity_dist   = 0.03*3500/pressure
        voxel_size        = 15
    
    elif (diffusion == "0.0percent"):
        Diff_smear        = 2.6
        energy_threshold  = 0.0004
        diff_scale_factor = 6
        radius_sf         = 7
        group_sf          = 3
        Tortuosity_dist   = 0.05*3500/pressure
        voxel_size        = 50

    elif (diffusion == "5percent" or diffusion == "5.0percent"): # because I messed up naming conventions

        voxel_size        = 10

        if (pressure == 1):
            Diff_smear        = 0.314/np.sqrt(pressure)
            energy_threshold  = 0.0002
            diff_scale_factor = 6
            radius_sf         = 7
            group_sf          = 5
            Tortuosity_dist   = 0.03*3500/pressure
        elif (pressure == 5):
            Diff_smear        = 0.314/np.sqrt(pressure)
            diff_scale_factor = 6
            energy_threshold  = 0.001
            radius_sf         = 7
            group_sf          = 30
            Tortuosity_dist   = 0.1*3500/pressure
        elif (pressure == 10):
            Diff_smear        = 0.314/np.sqrt(pressure)
            diff_scale_factor = 6
            energy_threshold  = 0.001
            radius_sf         = 7
            group_sf          = 30
            Tortuosity_dist   = 0.1*3500/pressure
        elif (pressure == 15):
            Diff_smear        = 0.314/np.sqrt(pressure)
            diff_scale_factor = 6
            energy_threshold  = 0.001
            radius_sf         = 7
            group_sf          = 30
            Tortuosity_dist   = 0.1*3500/pressure
        elif (pressure == 25):
            Diff_smear        = 0.314/np.sqrt(pressure)
            diff_scale_factor = 5
            energy_threshold  = 0.001
            radius_sf         = 30
            group_sf          = 3
            Tortuosity_dist   = 0.1*3500/pressure
        else:
            print("Unknown pressure configured")

    elif (diffusion == "next1t"): # this configures 15 bar fixed 3mm voxels for NEXT1t analysis
        Diff_smear        = 0.314/np.sqrt(pressure)
        diff_scale_factor = 4
        energy_threshold  = 0.001
        radius_sf         = 7
        group_sf          = 30
        Tortuosity_dist   = 0.1*3500/pressure
        det_half_length   = 1300
    else:
        print("Error gas percentage not defined at 60 V/cm/bar field")

    return Diff_smear, energy_threshold, diff_scale_factor, radius_sf, group_sf, Tortuosity_dist, voxel_size, det_half_length
# ---------------------------------------------------------------------------------------------------
# Function to check if any hit is outside a desired volume in cylinder geometry
# Returns False if the event is considered to be contained and True if contained
def CheckHitBounds(df, R, z_min, z_max):
    outside = (df.x**2 + df.y**2 > R**2) | (df.z < z_min) | (df.z > z_max)
    return not outside.any()
# ---------------------------------------------------------------------------------------------------
# Function to check if any hit is outside a desired volume in cylinder geometry
# Returns False if the event is considered to be contained and True if contained
# This function only applies for the NEXT-1t analysis. 15 bar 3mm vox fixed. 
def CheckHitBoundsNext1t(df):
    cube_size = 2600
    R = cube_size/2.0-20
    z1_max = cube_size/2.0-20
    z1_min = 20.1 # mm
    z2_max = -20.1 #mm
    z2_min = -cube_size/2.0+20
    outside = (df.x**2 + df.y**2 > R**2) | (df.z < z2_min) | (df.z > z1_max) | ((df.z-1300 > z2_max) & (df.z-1300 < z1_min)) # the -1300 shifts it back to zero to make this cut work
    return not outside.any()
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------