import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import itertools
import copy
from scipy.spatial.distance import cdist

colormap = plt.cm.get_cmap('Dark2')
color_cycle = itertools.cycle(colormap.colors)



# Function to add connections made
# current and current node index is input
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


# Function to check if a new connection would form a closed loop
def forms_cycle(node, target, connections_dict):

    query = node
    prev_node = node 
    # print(query)

    for index,n in enumerate(range(len(connections_dict))):
        
        # Get the connected nodes
        con_nodes = connections_dict[query]
        # print("Start",query, prev_node, con_nodes)

        # We hit a end-point and it didnt loop
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
    

# Helper function for testing for closed loops
def Testcycle(curr_node, conn_node,connected_nodes_, connections_, connection_count_):

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


def check_start_end_exists(number,Tracks):
    check_start = any(path["start"] == number for path in Tracks)
    check_end = any(path["end"] == number for path in Tracks)

    if (check_start or check_end):
        return True
    else:
        return False

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2['x'] - point1['x'])**2 + (point2['y'] - point1['y'])**2 + (point2['z'] - point1['z'])**2)

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


# Get the length and energy of a track
def GetMeanNodeDist(Tracks, data):

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


def GetTrackwithNode(closest_idx, Tracks_):
    for t in Tracks_:
        if (closest_idx in t["nodes"]):
            return t["id"]
    # The node wasnt found anywhere...
    return -1

def GetTrackDictwithNode(closest_idx, Tracks_):
    for t in Tracks_:
        if (closest_idx in t["nodes"]):
            return t
    # The node wasnt found anywhere...
    return -1

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


def GetUniqueTrackID(Tracks_):

    temp_track_id = -1

    for t in Tracks_:
        if temp_track_id <= t["id"]:
            temp_track_id = t["id"]

    # print("New Track ID is:",temp_track_id+1)

    return temp_track_id+1

# Any nodes without a connection can be re-added as a track
def AddConnectionlessNodes(connection_count, UpdatedTracks, data):

    for index, c in enumerate(connection_count):

        if (c == 0):
            hit_energy = data.iloc[index].energy
            Gamma = {"id":GetUniqueTrackID(UpdatedTracks), "start":index, "end":index, "nodes":[index], "length":0, "energy":hit_energy,"label":"gamma","c":"y"}
            UpdatedTracks.append(Gamma)


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

# Function to walk along a track segment till we get to an end
def GetNodePath(graph_, start_node, forward_node):
    
    graph = copy.deepcopy(graph_)
    
    path = [start_node]
    
    query = forward_node
    prev_node = start_node 

    for index,n in enumerate(range(len(graph))):

        path.append(query)
        
        # Get the connected nodes
        con_nodes = graph[query]

        # We hit a end-point and it didnt loop
        if (len(con_nodes) == 1):
            return path
        
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

            print("help!!")

        if (len(con_nodes) > 3 ):
            print("Error too many nodes in pathing that I was anticipating...")

        # Get the node that went in the query before
        if con_nodes[1] == prev_node:
            prev_node = query
            query = con_nodes[0]
        else:
            prev_node = query
            query = con_nodes[1]


# Function to calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Function to calculate the angle between two vectors
def angle_between_vectors(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Ensure the value is within [-1, 1] due to floating-point precision
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)


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
    mean_point = np.array([mean_x, mean_y, mean_z, energy_sum])

    all_visited = all_visited_ + list(closest_nodes)

    return mean_point, all_visited


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

    return pd.DataFrame(node_centers, columns=['x', 'y', 'z', 'energy'])



def RunClustering(node_centers_df, cluster_radii, binsize):

    event_id = node_centers_df.event_id.iloc[0]


    for R in cluster_radii:
        node_centers_df = Cluster(node_centers_df, R)

    node_centers_df["event_id"] = event_id

    # Create the bins ---- 
    xmin=-3000
    xmax=3000
    xbw=binsize

    ymin=-3000
    ymax=3000
    ybw=binsize

    zmin=0
    zmax=6000
    zbw=binsize

    # bw = 10 works well for co2 mix with 2 sigma diffusion

    # bins for x, y, z
    xbins = np.arange(xmin, xmax+xbw, xbw)
    ybins = np.arange(ymin, ymax+ybw, ybw)
    zbins = np.arange(zmin, zmax+zbw, zbw)

    # center bins for x, y, z
    xbin_c = xbins[:-1] + xbw / 2
    ybin_c = ybins[:-1] + ybw / 2
    zbin_c = zbins[:-1] + zbw / 2


    databin = node_centers_df.copy()

    # Now lets bin the data
    databin['x_smear'] = pd.cut(x=databin['x'], bins=xbins,labels=xbin_c, include_lowest=True)
    databin['y_smear'] = pd.cut(x=databin['y'], bins=ybins,labels=ybin_c, include_lowest=True)
    databin['z_smear'] = pd.cut(x=databin['z'], bins=zbins,labels=zbin_c, include_lowest=True)

    #Loop over the rows in the dataframe and merge the energies. Also change the bin center to use the mean x,y,z position
    x_mean_arr = []
    y_mean_arr = []
    z_mean_arr = []
    energy_mean_arr = []
    x_mean_arr_temp = np.array([])
    y_mean_arr_temp = np.array([])
    z_mean_arr_temp = np.array([])
    summed_energy = 0

    counter = 0

    # test_df = test_df.reset_index()
    databin = databin.sort_values(by=['x_smear', 'y_smear', 'z_smear'])


    for index, row in databin.iterrows():

        # First row
        if (counter == 0):
            temp_x = row["x_smear"]
            temp_y = row["y_smear"]
            temp_z = row["z_smear"]
            summed_energy +=row["energy"]
            event_id = row["event_id"]
            x_mean_arr_temp = np.append(x_mean_arr_temp, row["x"])
            y_mean_arr_temp = np.append(y_mean_arr_temp, row["y"])
            z_mean_arr_temp = np.append(z_mean_arr_temp, row["z"])
            counter+=1
            continue

        # Same bin
        if (row["x_smear"] == temp_x and row["y_smear"] == temp_y and row["z_smear"] == temp_z):
            x_mean_arr_temp = np.append(x_mean_arr_temp, row["x"])
            y_mean_arr_temp = np.append(y_mean_arr_temp, row["y"])
            z_mean_arr_temp = np.append(z_mean_arr_temp, row["z"])
            summed_energy +=row["energy"]

            # Final row
            if index == databin.index[-1]:
                if (summed_energy != 0): 
                    x_mean_arr = np.append(x_mean_arr,np.mean(x_mean_arr_temp))
                    y_mean_arr = np.append(y_mean_arr,np.mean(y_mean_arr_temp))
                    z_mean_arr = np.append(z_mean_arr,np.mean(z_mean_arr_temp))
                    energy_mean_arr.append(summed_energy)

        # Aggregate and store for next 
        else:
            if (summed_energy != 0): 
                x_mean_arr = np.append(x_mean_arr,np.mean(x_mean_arr_temp))
                y_mean_arr = np.append(y_mean_arr,np.mean(y_mean_arr_temp))
                z_mean_arr = np.append(z_mean_arr,np.mean(z_mean_arr_temp))
                energy_mean_arr.append(summed_energy)
            
            temp_x = row["x_smear"]
            temp_y = row["y_smear"]
            temp_z = row["z_smear"]
            summed_energy = 0
            x_mean_arr_temp = np.array([])
            y_mean_arr_temp = np.array([])
            z_mean_arr_temp = np.array([])
            
            
            x_mean_arr_temp = np.append(x_mean_arr_temp, row["x"])
            y_mean_arr_temp = np.append(y_mean_arr_temp, row["y"])
            z_mean_arr_temp = np.append(z_mean_arr_temp, row["z"])
            summed_energy +=row["energy"]

            # Final row
            if index == databin.index[-1]:
                if (summed_energy != 0): 
                    x_mean_arr = np.append(x_mean_arr,np.mean(x_mean_arr_temp))
                    y_mean_arr = np.append(y_mean_arr,np.mean(y_mean_arr_temp))
                    z_mean_arr = np.append(z_mean_arr,np.mean(z_mean_arr_temp))
                    energy_mean_arr.append(summed_energy)

        counter+=1

    # Make the dataframe again
    databin = pd.DataFrame({  "event_id" : event_id, "x" : x_mean_arr,  "y" : y_mean_arr,  "z" : z_mean_arr,  "energy" : energy_mean_arr  }) 

    databin["event_id"] = databin["event_id"].astype('int')

    return databin


# Function to plot connections
def plot_tracks(ax, x, y, connection_count, x_label, y_label, Tracks_):
    # Filter data for markers with count 1 or 0
    filtered_indices = [i for i, count in enumerate(connection_count) if count == 1 or count == 0 or count == 3]
    filtered_x = [x[i] for i in filtered_indices]
    filtered_y = [y[i] for i in filtered_indices]
    
    # # Define colors for filtered data
    colors = [None] * len(filtered_indices)
    for index, i in enumerate(filtered_indices):
        if connection_count[i] == 1:
            colors[index] = "r"
        elif (connection_count[i] == 0):
            colors[index] = "Orange"
        else:
            colors[index] = "DarkGreen"

    
    ax.scatter(filtered_x, filtered_y, c=colors, marker='o')

    # Plot connections
    for Track in Tracks_:
        for i, connection in enumerate(Track["nodes"]):
            if i == len(Track["nodes"]) - 1:
                break

            start_node = Track["nodes"][i]
            end_node = Track["nodes"][i + 1]

            ax.plot([x[start_node], x[end_node]],
                    [y[start_node], y[end_node]], color=Track["c"], linestyle="-")
            
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{x_label}-{y_label} Projection')


def MakeTracks(connection_count_, connected_nodes_, data_nodes, remaining_nodes, data, iteration, trk_ids, RebuiltTrack_):

    Track_arrays = []

    prim_track_id = -1
    prim_len = 0
    prim_track_arr = []
    prim_energy = 0

    # Get all nodes with single connections
    end_points = np.where(connection_count_ == 1)[0]
    end_points = [x for x in end_points if x in remaining_nodes]

    if (iteration == 0):
        primary_label = "Primary"
        delta_label = "Delta"
        color = "Teal"
    else:
        primary_label = "Brem"
        delta_label = "BremDelta"
        color = next(color_cycle)

    for index, end_point in enumerate(end_points):
        trkpath = GetLongestPath(connected_nodes_, end_point)
        Track_arrays.append(trkpath)

        trk_length = GetTrackLength(trkpath, data)

        if (trk_length > prim_len):
            prim_len = trk_length
            prim_track_id = index
            prim_track_arr = trkpath
            prim_energy = GetTrackEnergy(trkpath, data, False)
    
    # Create the primary track
    RebuiltTrack_.append({"id":trk_ids, "start":prim_track_arr[0], "end":prim_track_arr[-1], "length":trk_length, "energy":prim_energy, "label":primary_label, "c":color, "nodes":prim_track_arr})
    trk_ids = trk_ids + 1

    # Get all nodes with three connections in the primary track
    multi_connections = np.where(connection_count_ == 3)[0]
    prim_track_multi_connections = [x for x in multi_connections if x in prim_track_arr]

    for node in prim_track_multi_connections:
        delta_node = [x for x in connected_nodes_[node] if x not in prim_track_arr]

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
    single_points = [x for x in single_points if x in remaining_nodes]

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


# Function to walk along a track segment till we get to an end
def GetDeltaPath(graph_, start_node, forward_node, trkidx):
    
    graph = copy.deepcopy(graph_)
    
    paths = {trkidx : [start_node]}
    
    query = forward_node
    prev_node = start_node 

    for index,n in enumerate(range(len(graph))):

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

    return paths


def GetLongestPath(graph_, node):

    graph = copy.deepcopy(graph_)

    path = [node]
    
    query = graph[node][0] # The node should only have 1 connection
    prev_node = node 

    for index,n in enumerate(range(len(graph))):

        path.append(query)
        
        # Get the connected nodes
        con_nodes = graph[query]

        # We hit a end-point and it didnt loop
        if (len(con_nodes) == 1):
            return path
        
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

        # Get the node that went in the query before
        if con_nodes[1] == prev_node:
            prev_node = query
            query = con_nodes[0]
        else:
            prev_node = query
            query = con_nodes[1]

    return path



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


# Get the mean distances between nodes:
def GetMeanNodeDistances(df):

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

    return median_distance