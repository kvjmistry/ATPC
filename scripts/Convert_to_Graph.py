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
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm

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
    # xyz_mean = df[["x", "y", "z"]].mean()
    # xyz_std  = df[["x", "y", "z"]].std()
    # df[["x", "y", "z"]] = 0.5*(df[["x", "y", "z"]] - xyz_mean) / xyz_std
    
    # Normalize so that x,y,z are between 0 and 1
    df["z"] = df["z"]/6180.
    df["x"] = (df["x"]+6180./2.0)/6180.
    df["y"] = (df["y"]+6180./2.0)/6180.

    # Apply clipping to the energy
    df['energy'] = df['energy'].clip(upper=0.4)
    df['Tortuosity'] = df['Tortuosity'].clip(upper=5)
    
                                             # Min/Max
    df = MinMaxScale(df, "energy", 0, 0.4)   # 0, 0.4
    df = MinMaxScale(df, "Tortuosity", 1, 5) # 1, 5
    df = MinMaxScale(df, "angle", 0, 180)    # 0, 180

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
    df['label_cat'] = df['label_cat'].cat.set_categories(['Primary', 'Delta', 'Brem', 'BremDelta'], ordered=False )

    df['SubType'] = df['subType'].astype('category')
    df['SubType'] = df['SubType'].cat.set_categories(['0nubb', 'Bi', 'Tl', 'single'], ordered=False )

    # Integer encoding for training
    df['label_id'] = df['label_cat'].cat.codes
    df['SubType_cat'] = df['SubType'].cat.codes

    # Signal or background
    df["label"] = (df["Type"] == "0nubb").astype(int)

    return df
# ------------------------------------------------------------------------------
def event_to_track_graph(event_df, Track):
    
    # Reset the index
    event_df = event_df.reset_index(drop=True)
    
    x = torch.tensor(event_df[["x", "y", "z", "energy", "Tortuosity", "angle", "cum_dist_norm", "label_id"]].values, dtype=torch.float32) # (N,8): N rows of these features
    event_id = torch.tensor(event_df["event_id"].iloc[0])
    subType = torch.tensor(event_df["SubType_cat"].iloc[0])

    # Build the track
    edge_index, edge_attr = build_track_edges_with_attr(event_df, Track)
    
    y = torch.tensor([event_df["label"].iloc[0]], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, event_id=event_id, subType=subType)
# ------------------------------------------------------------------------------
def build_track_edges_with_attr(event_df, tracks):
    src = []
    dst = []
    edge_attr = []
    

    for t in tracks:
        nodes = t['nodes']
        for i in range(len(nodes) - 1):
            u = nodes[i]
            v = nodes[i + 1]

            # Distance between nodes
            v_pos = torch.tensor( event_df[event_df.id == v][["x", "y", "z"]].values[0], dtype=torch.float32 )
            u_pos = torch.tensor( event_df[event_df.id == u][["x", "y", "z"]].values[0], dtype=torch.float32 )
            d = v_pos - u_pos      # (dx, dy, dz)
            d_norm = torch.linalg.vector_norm(d)
            d_norm = torch.clamp(d_norm, max=0.2) # 0.2 comes from looking at the distribution
            d_norm = d_norm / 0.2

            src.append(u)
            dst.append(v)
            
            # Second column is flag to tell if we added a brem connection or not
            edge_attr.append(torch.tensor([d_norm, 0.0], dtype=torch.float))
    
    # Map the nodes id to a squential index in the dataframe
    id_map = {original_id: i for i, original_id in enumerate(event_df['id'])}
    src_indices = [id_map[id] for id in src]
    dst_indices = [id_map[id] for id in dst]
    
    # Add Brem Connections to the Primary Track
    # Uses sequential index from df, no need to remap these indices
    src_indices, dst_indices, edge_attr = AddBremConnection(event_df, tracks, src_indices, dst_indices, edge_attr)

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

        # All distances of each hits to track A to track B
        # A[:, None, :] expands A to (N,1,3)
        # B[None, :, :] expands B to (1,M,3)
        # Subtraction broadcasts them to a N x M x 3 matrix. 
        # Each cell (i,j) in this tensor contains the vector displacement (Δx,Δy,Δz)
        # between the i-th hit of Track A and the j-th hit of Track B.
        dist = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)

        # The index of global minimum
        iA, iB = np.unravel_index(dist.argmin(), dist.shape)

        # The actual DataFrame indices
        idx_A = df_track.index[iA]
        idx_B = df_primary.index[iB]
        min_distance = dist[iA, iB]
        
        min_distance = np.clip(min_distance, a_min = 0, a_max=0.2)
        min_distance = min_distance/0.2
        
        # Add bi-direction to these connections and angle of zero
        src_indices.append(idx_A)
        dst_indices.append(idx_B)
        src_indices.append(idx_B)
        dst_indices.append(idx_A)
        
        # Second column is flag to tell if we added a brem connection or not. Set to 1 here
        # Adding twice since we are doing bi-directional connection
        edge_attr.append(torch.tensor([min_distance, 1.0], dtype=torch.float))
        edge_attr.append(torch.tensor([min_distance, 1.0], dtype=torch.float))
    
    return src_indices, dst_indices, edge_attr
# ------------------------------------------------------------------------------
# def build_graph_dataset(df, Tracks):
#     graphs = []

#     for i, ev_id in enumerate(df.event_id.unique()):
#         if (i % 100 == 0):
#             print("On event", i, "/", len(df.event_id.unique()))
#         graphs.append(event_to_track_graph(df[df.event_id == ev_id], Tracks[ev_id])) # Track connections
#     return graphs
def build_graph_dataset(df, Tracks, n_jobs=60):
    unique_event_ids = df.event_id.unique()
    
    # We wrap the function call in delayed()
    graphs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(event_to_track_graph)(
            df[df.event_id == ev_id], 
            Tracks[ev_id]
        ) 
        for ev_id in tqdm(unique_event_ids, desc="Processing Events")
    )
    
    return graphs
# ------------------------------------------------------------------------------
def GetGraphs(event_list, jobid, splitsize):
    
    tot_sig_events    = 0
    tot_Bi_events     = 0
    tot_Tl_events     = 0
    tot_single_events = 0
    
    basepath = "/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples/"
    
    num_events_nubb, data_nubb = LoadData(f"{basepath}/ATPC_0nubb/1bar/5percent/reco/*.h5",   "0nubb",  event_list, splitsize, jobid)
    Tracks_nubb,  _,  _        = LoadPickle(f"{basepath}/ATPC_0nubb/1bar/5percent/pkl/*.pkl", "0nubb",  event_list, splitsize, jobid)
    
    num_events_Bi, data_Bi = LoadData(f"{basepath}/ATPC_Bi_ion/1bar/5percent/reco/*.h5",   "Bi",  event_list, splitsize, jobid)
    Tracks_Bi,  _,  _      = LoadPickle(f"{basepath}/ATPC_Bi_ion/1bar/5percent/pkl/*.pkl", "Bi",  event_list, splitsize, jobid)
    
    num_events_Tl, data_Tl = LoadData(f"{basepath}/ATPC_Tl_ion/1bar/5percent/reco/*.h5",   "Tl",  event_list, splitsize, jobid)
    Tracks_Tl,  _,  _      = LoadPickle(f"{basepath}/ATPC_Tl_ion/1bar/5percent/pkl/*.pkl", "Tl",  event_list, splitsize, jobid)
    
    num_events_single, data_single = LoadData(f"{basepath}/ATPC_single/1bar/5percent/reco/*.h5",   "single",  event_list, splitsize, jobid)
    Tracks_single,  _,  _          = LoadPickle(f"{basepath}/ATPC_single/1bar/5percent/pkl/*.pkl", "single",  event_list, splitsize, jobid)
    
    tot_sig_events+=num_events_nubb
    tot_Bi_events+=num_events_Bi
    tot_Tl_events+=num_events_Tl
    tot_single_events+=num_events_single
    
    print("Adding Track Info for chunk:", jobid)
    data_track_nubb   = AddTrackInfo(data_nubb, Tracks_nubb)
    data_track_Bi     = AddTrackInfo(data_Bi, Tracks_Bi)
    data_track_Tl     = AddTrackInfo(data_Tl, Tracks_Tl)
    data_track_single = AddTrackInfo(data_single, Tracks_single)
    
    # Normalize
    print("Normalizing Data")
    data_track_nubb   = NormlizeDataFrame(data_track_nubb)
    data_track_Bi     = NormlizeDataFrame(data_track_Bi)
    data_track_Tl     = NormlizeDataFrame(data_track_Tl)
    data_track_single = NormlizeDataFrame(data_track_single)
    
    print("Adding Categorical data")
    data_track_nubb   = AddCategories(data_track_nubb)
    data_track_Bi     = AddCategories(data_track_Bi)
    data_track_Tl     = AddCategories(data_track_Tl)
    data_track_single = AddCategories(data_track_single)
    
    print("Building Graphs for chunk:", jobid)
    graphs_nubb   = build_graph_dataset(data_track_nubb, Tracks_nubb)
    graphs_Bi     = build_graph_dataset(data_track_Bi, Tracks_Bi)
    graphs_Tl     = build_graph_dataset(data_track_Tl, Tracks_Tl)
    graphs_single = build_graph_dataset(data_track_single, Tracks_single)
    
    # Combine all the graphs together
    all_graphs = graphs_nubb + graphs_Bi + graphs_Tl + graphs_single
    
    # Get the label to split events up with 
    subtype_lists = []
    for g in all_graphs:
        subtype_lists.append(g.subType.item())

    # Split the graphs up
    print("Splitting the graphs up")
    graphs_tmp, graphs_test, subtype_tmp, subtype_test   = train_test_split(all_graphs, subtype_lists, test_size=0.10, stratify=subtype_lists, random_state=jobid)
    graphs_train, graphs_val, subtype_train, subtype_val = train_test_split(graphs_tmp, subtype_tmp, test_size=2/9,   stratify=subtype_tmp,   random_state=jobid)

    if (len(event_list) != 0):
        print("Saving graph to ", f'{basepath}/GNN_files_MLP/ATPC_GNN_chunk_[train/val/test]_{jobid}.pt')
        torch.save(graphs_test,   f'{basepath}/GNN_files_MLP/ATPC_GNN_chunk_test_{jobid}.pt')
        torch.save(graphs_val,    f'{basepath}/GNN_files_MLP/ATPC_GNN_chunk_val_{jobid}.pt')
        torch.save(graphs_train,  f'{basepath}/GNN_files_MLP/ATPC_GNN_chunk_train_{jobid}.pt')
        
    else:
        print("Saving graph to ", f'{basepath}/GNN_files/ATPC_GNN_chunk_[train/val/test]_{jobid}.pt')
        torch.save(graphs_test,   f'{basepath}/GNN_files/ATPC_GNN_chunk_test_{jobid}.pt')
        torch.save(graphs_val,    f'{basepath}/GNN_files/ATPC_GNN_chunk_val_{jobid}.pt')
        torch.save(graphs_train,  f'{basepath}/GNN_files/ATPC_GNN_chunk_train_{jobid}.pt')
        
    print("tot_sig_events saved:", tot_sig_events)
    print("tot_bkg_events saved:", tot_Bi_events + tot_Tl_events + tot_single_events)
    print("-------")
    print("tot_Bi_events saved:", tot_Bi_events)
    print("tot_Tl_events saved:", tot_Tl_events)
    print("tot_single_events saved:", tot_single_events)
# ------------------------------------------------------------------------------

listin=f"../eventlists/ATPC_1bar_5percent_highstats.csv"

# The job id

# Job id needs to range from 1 to splitsize
jobid = int(sys.argv[1])
splitsize=int(sys.argv[2])
filter_events=False

print("JobID/splitsize:", jobid, "/", splitsize)

event_list = []

if filter_events:
    event_list = pd.read_csv(listin);


GetGraphs(event_list, jobid, splitsize)