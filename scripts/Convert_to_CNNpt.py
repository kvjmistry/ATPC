# This script will convert the track reco files to a graph for ML training. 
# This is for using with a slurm script

# This notebook produces track objects by looping over the signal and background samples in segements

import pandas as pd
import numpy as np
import math
import glob
import sys
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

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
class EventDataset(Dataset):
    def __init__(self, event_dfs_list, labels, voxel_size, spatial_shape):
        self.event_dfs_list = event_dfs_list
        self.labels = labels
        self.voxel_size = voxel_size
        self.spatial_shape = spatial_shape

        print("Voxel Size:", self.voxel_size, "mm", "| Voxel Grid Size:", self.spatial_shape, "| Total Events:", len(self.labels), "\n")

    def __getitem__(self, idx):
        
        # Grab the df for index 
        event_df = self.event_dfs_list[idx]
        coords, feats = voxelize_event(event_df, self.voxel_size)
        label = self.labels[idx]
        
        event_id = event_df.event_id.iloc[0] 
        subType = event_df['subType'].iloc[0]
        
        meta = {"event_id": event_id, "subType": subType}
        
        # Convert to torch objects
        coords = torch.from_numpy(coords).int()
        feats  = torch.from_numpy(feats).float()
        label  = torch.tensor(self.labels[idx])
        
        return coords, feats, label, meta

    def __len__(self):
        return len(self.event_dfs_list)
# ------------------------------------------------------------------------------
# Function to voxelize the event
# always set idx = 0, as collate_fn will overwrite this with the local batch
# voxel size in mm
def voxelize_event(event_df, VOXEL_SIZE):
    
    # Convert the coorinates into integers
    event_df = event_df.copy()
    event_df['z_int'] = np.floor(event_df['z'] / VOXEL_SIZE).astype(np.int32)
    event_df['y_int'] = np.floor(event_df['y'] / VOXEL_SIZE).astype(np.int32)
    event_df['x_int'] = np.floor(event_df['x'] / VOXEL_SIZE).astype(np.int32)

    # In case of duplicates, we need to aggregrate them
    voxel_df = event_df.groupby(['z_int', 'y_int', 'x_int']).agg({'energy': 'sum' }).reset_index()

    # Shift to start at 0
    coords = voxel_df[['z_int', 'y_int', 'x_int']].values
    # coords -= coords.min(axis=0)
    
    # Spconv needs first column to be the batch index
    batch_indices = np.full((len(voxel_df), 1), 0, dtype=np.int32)
    spconv_coords = np.hstack([batch_indices, coords])
    
    features = voxel_df[['energy']].values.astype(np.float32)

    return spconv_coords, features
# ------------------------------------------------------------------------------
# Creates a list of dataframes, one for each event, baed on test-train split
def make_event_df_list(df, event_ids):
    grouped = df.groupby('event_id')
    
    # Gets the whole dataframe for each eventid
    event_dfs_list = [grouped.get_group(eid).copy() for eid in event_ids]
    
    # Gets the labels for each event in the df
    labels = [g['label'].iloc[0] for g in event_dfs_list]
    return event_dfs_list, labels
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def GetSpatialShape(df, VOXEL_SIZE):
    # Estimate the max grid size based on the voxel size used
    max_z = max(df['z'])
    max_y = max(df['y'])
    max_x = max(df['x'])

    # Convert to voxel units and add a buffer
    global_max_coords = np.array([max_z, max_y, max_x])
    spatial_shape = np.ceil(global_max_coords / VOXEL_SIZE).astype(int) + 1

    # Round up to a multiple of 8 or 16
    # This ensures that stride-2 layers divide cleanly
    input_data_shape = [((s + 15) // 16) * 16 for s in spatial_shape]
    
    return input_data_shape
# ------------------------------------------------------------------------------
def MinMaxScale(df, label, var_min, var_max):
    # Min-Max scaling
    df[label] = (df[label] - var_min) / (var_max - var_min)
    return df
# ------------------------------------------------------------------------------
def ApplyScaling(df):
    # Normalize so that x,y,z are between 0 and 1
    # df["z"] = df["z"]/6180.
    df["x"] = (df["x"]+6180./2.0)
    df["y"] = (df["y"]+6180./2.0)

    # Apply clipping to the energy
    df['energy'] = df['energy'].clip(upper=0.4)

    # Min/Max
    df = MinMaxScale(df, "energy", 0, 0.4)   # 0, 0.4
    
    return df

# ------------------------------------------------------------------------------
# Load in the MC true for all events
def process_single_file(f):
    """Worker function for one file"""
    try:
        hits = pd.read_hdf(f, "MC/hits")
        # hits = pd.read_hdf(f, "data")
        return hits
    except Exception as e:
        print(f"Error in {f}: {e}")
        return None
# ------------------------------------------------------------------------------
def LoadFilesParallel(files):
    
    # n_jobs=-1 uses all available cores (all 60)
    # prefer="threads" is good for I/O, but "processes" is better for pandas filtering
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_file)(f) for f in files
    )
    
    # Filter out None results if any files failed to load
    results = [res for res in results if res is not None]
    
    return pd.concat(results)

# ------------------------------------------------------------------------------
# Get the data
def LoadData(f_nubb, f_Bi, f_Tl, f_single):

    folder="raw"
    # folder="reco"

    nubb = LoadFilesParallel(f_nubb)
    nubb["Type"] = "0nubb"
    nubb["subType"] = "0nubb"

    Bi = LoadFilesParallel(f_Bi)
    Bi["subType"] = "Bi"

    Tl = LoadFilesParallel(f_Tl)
    Tl["subType"] = "Tl"

    single = LoadFilesParallel(f_single)
    single["subType"] = "single"

    Bkg = pd.concat([Bi, Tl, single])
    Bkg["Type"] = "Bkg"


    df = pd.concat([nubb, Bkg])
    df = df.reset_index(drop=True)

    df = df[["event_id", "x", "y", "z", "energy", "Type", "subType"]]
    return df
# ------------------------------------------------------------------------------

basepath = "/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples/"

VOXEL_SIZE=8.0

# The job id

# Job id needs to range from 1 to splitsize
jobid = int(sys.argv[1])
splitsize=int(sys.argv[2])

print("JobID/splitsize:", jobid, "/", splitsize)


# Get all file names for each event category
f_nubb   = get_file_chunk(f"{basepath}/ATPC_0nubb/1bar/5percent/raw/*.h5",  splitsize, jobid)
f_Bi     = get_file_chunk(f"{basepath}/ATPC_Bi_ion/1bar/5percent/raw/*.h5", splitsize, jobid)
f_Tl     = get_file_chunk(f"{basepath}/ATPC_Tl_ion/1bar/5percent/raw/*.h5", splitsize, jobid)
f_single = get_file_chunk(f"{basepath}/ATPC_single/1bar/5percent/raw/*.h5", splitsize, jobid)

print("f_nubb", len(f_nubb))
print("f_Bi", len(f_Bi))
print("f_Tl", len(f_Tl))
print("f_single", len(f_single))

# Load the data
df = LoadData(f_nubb, f_Bi, f_Tl, f_single)
print(df)

# Apply scaling
df = ApplyScaling(df)

# Get the spatial shape
input_data_shape = GetSpatialShape(df, VOXEL_SIZE)

# Type: "0nubb" or "Bkg"
df['label'] = (df['Type'] == "0nubb").astype(int)

# Event-level labels
event_labels = df.groupby('event_id')['label'].first()
event_ids    = event_labels.index.values
event_y      = event_labels.values

# Split 70/20/10
# We are splitting the dataset by the event ids
ev_tmp, ev_test, y_tmp, y_test   = train_test_split(event_ids, event_y, test_size=0.10, stratify=event_y, random_state=jobid)
ev_train, ev_val, y_train, y_val = train_test_split(ev_tmp, y_tmp, test_size=2/9, stratify=y_tmp, random_state=jobid)

train_events_df_list, train_labels = make_event_df_list(df, ev_train)
val_events_df_list,   val_labels   = make_event_df_list(df, ev_val)
test_events_df_list,  test_labels  = make_event_df_list(df, ev_test)

train_dataset = EventDataset(train_events_df_list, train_labels, VOXEL_SIZE, input_data_shape)
val_dataset   = EventDataset(val_events_df_list,   val_labels,   VOXEL_SIZE, input_data_shape)
test_dataset  = EventDataset(test_events_df_list,  test_labels,  VOXEL_SIZE, input_data_shape)

print("Saving dataset to ", f'{basepath}/CNN_files/ATPC_CNN_chunk_[train/val/test]_{jobid}.pt')
torch.save(train_dataset,   f'{basepath}/CNN_files/ATPC_CNN_chunk_train_{jobid}.pt')
torch.save(val_dataset,     f'{basepath}/CNN_files/ATPC_CNN_chunk_val_{jobid}.pt')
torch.save(test_dataset,    f'{basepath}/CNN_files/ATPC_CNN_chunk_test_{jobid}.pt')