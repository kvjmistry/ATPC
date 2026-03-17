# This script will convert the track reco files to a graph for ML training. 
# This is for using with a slurm script

# This notebook produces track objects by looping over the signal and background samples in segements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import glob
import sys
import torch
from torch_geometric.data import Data

# ------------------------------------------------------------------------------
# Get a subset of the total files to iterate over
def get_file_chunk(path, splitsize, chunk):
    
    files = sorted(glob.glob(path))
    total_files = len(files)
    
    chunk_size = math.ceil(total_files / splitsize)
    
    start = (chunk - 1) * chunk_size
    end = start + chunk_size
    
    return files[start:end]
# ------------------------------------------------------------------------------
def LoadData(path, subType, event_list, splitsize, chunk):
    
    # print("On Chunk", chunk, "/", splitsize)
    
    # Choose whether to filter or not based on the length of the event list
    filter_events = False
    if (len(event_list) != 0):
        filter_events = True
    
    # Get a subset of all the files
    files = get_file_chunk(path, splitsize, chunk)
    
    # This is a list of selected events
    if (filter_events):
        event_list = event_list[event_list.subType.str.contains(subType)].event_id.values
    
    data_all = []
    
    for i, f in enumerate(files):
        # if i %50 ==0:
        #     print(f"{i} /", len(files))
            
        data = pd.read_hdf(f, "data")
        
        if (filter_events):
            data  = data[data.event_id.isin(event_list)] # filter events
        
        data["subType"] = subType
        if (subType == "0nubb"):
            data["Type"] = "0nubb"
        else:
            data["Type"] = "Bkg"
            
        data_all.append(data)
    
    data_all = pd.concat(data_all)
    num_events = len(data_all.event_id.unique())
    return num_events, data_all
# ------------------------------------------------------------------------------
def LoadPickle(path, subType, event_list, splitsize, chunk):
    
    # print("On Chunk", chunk, "/", splitsize)
    
    # Choose whether to filter or not based on the length of the event list
    filter_events = False
    if (len(event_list) != 0):
        filter_events = True
    
    # Get a subset of all the files
    files = get_file_chunk(path, splitsize, chunk)
    
    if (filter_events):
        event_list = event_list[event_list.subType.str.contains(subType)].event_id.values
    
    Tracks = {}
    connections = {}
    connection_counts = {}
    counter = 0
    for index, f in enumerate(files):

        # if index %50 ==0:
        #     print(f"{index} /", len(files))

        with open(f, 'rb') as pickle_file:  # Use 'rb' for reading in binary

            # Load data from file
            try:
                with open(f, 'rb') as pickle_file:
                    track_data = pickle.load(pickle_file)
                    conn_data = pickle.load(pickle_file)
                    conn_count_data = pickle.load(pickle_file)
                    
            except Exception as e:
                print(f"An unexpected error occurred while loading '{f}': {e}")
                continue
            
            if (filter_events):
                # Filter during loading
                track_data = {k: v for k, v in track_data.items() if k in event_list}
                conn_data = {k: v for k, v in conn_data.items() if k in event_list}
                conn_count_data = {k: v for k, v in conn_count_data.items() if k in event_list}
            
            counter+=len(track_data)
            
            # Initialize or update dictionaries
            if index == 0:
                Tracks = track_data
                connections = conn_data
                connection_counts = conn_count_data
            else:
                Tracks.update(track_data)
                connections.update(conn_data)
                connection_counts.update(conn_count_data)
                
    return Tracks, connections, connection_counts
# ------------------------------------------------------------------------------
# Track info to the data table
def AddTrackInfo(df, Tracks):

    rows = []
    for event_id, track_list in Tracks.items():
        for t in track_list:
            for node in t['nodes']:
                rows.append({'event_id': event_id,'id': node,'label': t['label']})
                
    tracks_df = pd.DataFrame(rows)
    df_merged = df.merge(tracks_df,how='inner',on=['event_id', 'id'])
    
    return df_merged
# ------------------------------------------------------------------------------
def MinMaxScale(df, label, var_min, var_max):
    # Min-Max scaling
    df[label] = (df[label] - var_min) / (var_max - var_min)
    return df
# ------------------------------------------------------------------------------
# Function for Min-Max normalization
def normalize_group(group):
    # Avoid division by zero if a group has only one point or max == min
    if group.max() == group.min():
        return group * 0.0
    return (group - group.min()) / (group.max() - group.min())
# ------------------------------------------------------------------------------
def NormlizeDataFrame(df):

    # Normalize the columns
    xyz_mean = df[["x", "y", "z"]].mean()
    xyz_std  = df[["x", "y", "z"]].std()
    df[["x", "y", "z"]] = 0.5*(df[["x", "y", "z"]] - xyz_mean) / xyz_std

    # Apply clipping to the energy
    df['energy'] = df['energy'].clip(upper=0.4)
    df['Tortuosity'] = df['Tortuosity'].clip(upper=5)

    df = MinMaxScale(df, "energy", 0, 0.4) # 0, 0.4
    df = MinMaxScale(df, "Tortuosity", 1, 5) # 1, 4
    df = MinMaxScale(df, "angle", 0, 180) # 0, 180

    # Normalize the cumulative distance per event and track id
    df['cum_dist_norm'] = df.groupby(['event_id','trkID'])['cumulative_distance'].transform(normalize_group)
    
    return df
# ------------------------------------------------------------------------------
def AddCategories(df):

  # Convert the label category to a trainable parameter
  df['label_cat'] = (
      df['label']
        .str.replace(r'^Delta\d+$', 'Delta', regex=True)
        .str.replace(r'^BremDelta\d+$', 'BremDelta', regex=True)
  )

  df['label_cat'] = df['label_cat'].astype('category')
  df['label_cat'] = df['label_cat'].cat.set_categories(['Primary', 'Delta', 'Brem', 'BremDelta'],ordered=False )

  df['SubType'] = df['subType'].astype('category')
  df['SubType'] = df['SubType'].cat.set_categories(['0nubb', 'Bi', 'Tl', 'single'],ordered=False )

  # Integer encoding for training
  df['label_id'] = df['label_cat'].cat.codes
  df['SubType_cat'] = df['SubType'].cat.codes

  label_map = {"0nubb": 1, "Bkg": 0}
  df["label"] = (df["Type"] == "0nubb").astype(int)
  
  return df
# ------------------------------------------------------------------------------
def event_to_track_graph(event_df, Track):
    
    # Reset the index
    event_df = event_df.reset_index(drop=True)
    
    pos = torch.tensor(
        event_df[["x", "y", "z"]].values,
        dtype=torch.float32
    ) # N rows of [x,y,z]
    
    x = torch.tensor(event_df[["z", "energy", "Tortuosity", "cum_dist_norm", "label_id"]].values, dtype=torch.float32) # (N,5): N rows of these features
    event_id = torch.tensor(event_df["event_id"].iloc[0])
    subType = torch.tensor(event_df["SubType_cat"].iloc[0])

    # Build the track
    edge_index, edge_attr = build_track_edges_with_attr(event_df, Track, pos)
    
    y = torch.tensor([event_df["label"].iloc[0]],dtype=torch.long)
    
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, y=y, event_id=event_id, subType=subType)
# ------------------------------------------------------------------------------
def build_track_edges_with_attr(event_df, tracks, pos):
    src = []
    dst = []
    edge_attr = []
    prev_dir = None  # for angle computation

    for t in tracks:
        nodes = t['nodes']
        for i in range(len(nodes) - 1):
            u = nodes[i]
            v = nodes[i + 1]

            # displacement vector
            d = pos[v] - pos[u]      # (dx, dy, dz)
            d_norm = d / 10.         # Divide by 10 to get a scale closer to ~1
            dist = torch.norm(d)     # |d|

            # direction / angle
            direction = d / (dist + 1e-8)
            
            # angle wrt previous segment
            if prev_dir is None:
                theta_norm = torch.tensor(0.0)
            else:
                cos_theta = torch.dot(prev_dir, direction)
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                theta = torch.acos(cos_theta)
                theta_norm = theta / math.pi # normalize angle
            
            prev_dir = direction

            src.append(u)
            dst.append(v)
            
            # Third column is flag to tell if we added a brem connection or not
            edge_attr.append(torch.tensor([dist, 0.0], dtype=torch.float))
    
    # Map the nodes "id" to index in the dataframe
    id_map = {original_id: i for i, original_id in enumerate(event_df['id'])}
    src_indices = [id_map[id] for id in src]
    dst_indices = [id_map[id] for id in dst]
    
    # Add Brem Connections to the Primary Track
    # Uses sequential index from df, no need to remap these indices
    src_indices, dst_indices, edge_attr = AddBremConnection(event_df, tracks, src_indices, dst_indices, edge_attr)

    # edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_attr = torch.stack(edge_attr)

    return edge_index, edge_attr
# ------------------------------------------------------------------------------
# Need to add an edge connection between brems and the primary track
def AddBremConnection(event_df, tracks, src_indices, dst_indices, edge_attr):
    
    # Get the primary track
    for t in tracks:
        if (t["label"] == "Primary"):
            df_primary = event_df[event_df.id.isin(t["nodes"])]
            
    for t in tracks:
        
        # Skip making connections with deltas
        if ("Delta" in t["label"] or "Primary" in t["label"] ):
            continue
        
        df_track = event_df[event_df.id.isin(t["nodes"])]
        
        A = df_track[["x", "y", "z"]].to_numpy()
        B = df_primary[["x", "y", "z"]].to_numpy()

        dist = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

        # index of global minimum
        iA, iB = np.unravel_index(dist.argmin(), dist.shape)

        # actual DataFrame indices
        idx_A = df_track.index[iA]
        idx_B = df_primary.index[iB]
        min_distance = dist[iA, iB]
        
        # Add bi-direction to these connections and angle of zero
        src_indices.append(idx_A)
        dst_indices.append(idx_B)
        
        # Third column is flag to tell if we added a brem connection or not. Set to 1 here
        edge_attr.append(torch.tensor([min_distance, 1.0], dtype=torch.float))

        src_indices.append(idx_A)
        dst_indices.append(idx_B)
        edge_attr.append(torch.tensor([min_distance, 1.0], dtype=torch.float))
        
    
    return src_indices, dst_indices, edge_attr
# ------------------------------------------------------------------------------
def build_graph_dataset(df, Tracks):
    graphs = []

    for ev_id in df.event_id.unique():
        graphs.append(event_to_track_graph(df[df.event_id == ev_id], Tracks[ev_id])) # Track connections
    return graphs
# ------------------------------------------------------------------------------
def GetGraphs(event_list, mode, jobid, splitsize):
    print("Getting Graphs for:", mode)
    
    tot_events = 0
    
    if mode == "Bi" or mode == "Tl":
        basepath = f"/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples/ATPC_{mode}_ion/1bar/5percent/"
    else:
        basepath = f"/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples/ATPC_{mode}/1bar/5percent/"
        
    num_events, data     = LoadData(f"{basepath}/reco/*.h5", mode,  event_list, splitsize, jobid)
    Tracks,  connections,  connection_counts  = LoadPickle(f"{basepath}/pkl/*.pkl", mode,  event_list, splitsize, jobid)
    
    tot_events+=num_events
        
    print("Adding Track Info for chunk:", jobid)
    df_merged = AddTrackInfo(data, Tracks)
    df_merged = NormlizeDataFrame(df_merged)
    df_merged = AddCategories(df_merged)
    print("Building Graph for chunk:", jobid)
    graphs = build_graph_dataset(df_merged, Tracks)
    print("Saving graph to ", f'{basepath}/GNN_files/ATPC_GNN_{mode}_chunk_{jobid}.pt')
    torch.save(graphs, f'{basepath}/GNN_files/ATPC_GNN_{mode}_chunk_{jobid}.pt')
        
    print("tot_events saved:", tot_events)
# ------------------------------------------------------------------------------


# We have ~5k files in a set, use 40 nodes at a time

basepath = "/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples/"

listin=f"../eventlists/ATPC_1bar_5percent_highstats.csv"

# The job id

# Job id needs to range from 1 to splitsize
jobid = int(sys.argv[1])
splitsize=int(sys.argv[2])
filter_events=False

# options are 0nubb, Bi, Tl, single
mode = sys.argv[3]


print("JobID/splitsize:", jobid, "/", splitsize)
print("Mode:", mode)

event_list = []

if filter_events:
    event_list = pd.read_csv(listin);

GetGraphs(event_list, mode, jobid, splitsize)