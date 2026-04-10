#!/usr/bin/env python

import pandas as pd
import numpy as np

import pickle
import glob
import sys
sys.path.append("../scripts")
from TrackReconstruction_functions import *
from collections import Counter
import json
import os
import time

from tqdm import tqdm
import lmdb
import pickle

# For profiling
import torch.cuda.nvtx as torch_nvtx

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, TransformerConv
from torch_geometric.data import Data

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from joblib import Parallel, delayed


# Check for CUDA (NVIDIA) or MPS (Apple)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

scale_factor_bkg = None  # Initialize it globally or at the top
mapped_dataset=False # don't change this!
df_merged = pd.DataFrame()
mode = "GraphTransformer" # Graph Transformer Network

class LMDBEventDataset(Dataset):
    def __init__(self, lmdb_path, data_sample, one_vs_all=False):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.one_vs_all = one_vs_all

        # Load in the metadata
        meta_file = f"/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//GNN_files/GNN_{data_sample}_meta.json"
        # meta_file = f"GNN_{data_sample}_meta_test.json"
        
        # Define mapping from subtype to the right classes
        self.multi_map = {0: 3, 1: 0, 2: 1, 3: 2} # transforms binary to multiclass during training based on the subtype stored
        
        print(f"Loading metadata from cache: {meta_file}")
        with open(meta_file, 'r') as f:
            cache_data = json.load(f)
            raw_counts = cache_data['subtype_counts']
            self.subtype_counts = Counter({int(k): v for k, v in raw_counts.items()})
            
        print("The ova is set to ", self.one_vs_all)
        self.print_scale_factors(data_sample)

        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f"{idx:09d}".encode()
            graph = pickle.loads(txn.get(key))
            
            # If one vs all mode, then we remap the data, else leave it alone as its mapped correctly
            sub_type = int(graph.subType)
            
            if self.one_vs_all:
                new_y = self.multi_map.get(sub_type, 0)
                graph.y = torch.tensor([new_y], dtype=torch.long)

        return graph
        
    def print_scale_factors(self, data_sample):
        sig = self.subtype_counts[0]
        bi  = self.subtype_counts[1]
        tl  = self.subtype_counts[2]
        sn  = self.subtype_counts[3]
        total_bkg = bi + tl + sn

        print(f"\n--- {data_sample} Dataset Summary ---")
        print(f"Signal (0nubb): {sig}")
        print(f"Background: {total_bkg}")
        print(f"Thalium: {tl}")
        print(f"Bismuth: {bi}")
        print(f"Single: {sn}")

one_vs_all = False
train_dataset = LMDBEventDataset("/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//GNN_files/GNN_Train_events.lmdb", "Train", one_vs_all)
val_dataset   = LMDBEventDataset("/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//GNN_files/GNN_Validation_events.lmdb", "Validation", one_vs_all)
test_dataset  = LMDBEventDataset("/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//GNN_files/GNN_Test_events.lmdb", "Test", one_vs_all)


BATCH_SIZE=600
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=4)


# Define the GNN
class EventTransformerConv(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_node_features, edge_dim):
        super().__init__()
        self.conv1 = TransformerConv(num_node_features, hidden_channels, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x



# Dervive weights based on inverse frequency
# Total / (Num_Classes * Class_Count)
# This method boosts signal and scales background down simultaniously
# average weight will remain 1
# Grab counts from the dataset object
sig = train_dataset.subtype_counts[0]
bi  = train_dataset.subtype_counts[1]
tl  = train_dataset.subtype_counts[2]
sn  = train_dataset.subtype_counts[3]
total_bkg = bi + tl + sn

# Calculate the raw factors
sf_bkg    = sig / total_bkg if total_bkg > 0 else 1.0
sf_bi     = sig / bi if bi > 0 else 1.0
sf_tl     = sig / tl if tl > 0 else 1.0
sf_single = sig / sn if sn > 0 else 1.0

total = total_bkg + sig

if one_vs_all:
    num_classes_train = 4
    extension = "ova"
    # Order: [Bi, Tl, Single, 0nubb]
    weight_bi = total / (num_classes_train * bi)
    weight_tl = total / (num_classes_train * tl)
    weight_single = total / (num_classes_train * sn)
    weight_signal = total / (num_classes_train * sig)

    train_weights = torch.tensor([weight_bi, weight_tl, weight_single, weight_signal]).to(device)
    print("Bi, Tl, Single, 0nubb")
    print(train_weights)
else:
    num_classes_train = 2
    extension = "binary"
    # Order: [Background, Signal]
    
    weight_bkg = total / (num_classes_train * total_bkg)
    weight_signal = total / (num_classes_train * sig)
    train_weights = torch.tensor([weight_bkg, weight_signal]).to(device)
    print("Background, Signal")
    print(train_weights)


# Function to get the actual background rejection factors for performance
def evaluate_physics_metrics(y_true, p_signal, subtypes, target_eff=0.60):

    fpr, tpr, thresholds = roc_curve(y_true, p_signal)
    
    # Find the global threshold where Signal Efficiency (TPR) is closest to target efficiency
    idx = np.where(tpr >= target_eff)[0][0]
    cut_value = thresholds[idx]
    actual_sig_eff = tpr[idx]
    
    # Total Background Rejection Factor
    total_bkg_acceptance = fpr[idx] 
    total_rejection_factor = 1.0 / total_bkg_acceptance if total_bkg_acceptance > 0 else float('inf')
    
    # Calculate Rejection Factor for each Subtype
    bkg_mask = (y_true == 0)
    bkg_p_signal = p_signal[bkg_mask]
    bkg_subtypes = subtypes[bkg_mask]
    
    subtype_rejection_factors = {}
    unique_subtypes = np.unique(bkg_subtypes)
    
    for subtype in unique_subtypes:
        # Get probabilities just for this specific background type
        subtype_mask = (bkg_subtypes == subtype)
        total_subtype_events = np.sum(subtype_mask)
        
        if total_subtype_events == 0:
            continue
            
        # How many of this subtype falsely passed the signal cut?
        subtype_false_positives = np.sum(bkg_p_signal >= cut_value)
        
        # Subtype Acceptance = Passed / Total
        subtype_acceptance = subtype_false_positives / total_subtype_events
        
        # Subtype Rejection Factor = 1 / Acceptance
        if subtype_acceptance > 0:
            subtype_rej = 1.0 / subtype_acceptance
        else:
            subtype_rej = float('inf')
            
        subtype_rejection_factors[subtype] = subtype_rej

    return actual_sig_eff, total_rejection_factor, subtype_rejection_factors


# mode = "GAT" # Graph Attention Network
mode = "GraphTransformer" # Graph Transformer Network


# Define your ML models here
model = EventTransformerConv(hidden_channels=64, num_classes=num_classes_train, num_node_features=7, edge_dim=2).to(device) # Transformer model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
criterion = nn.CrossEntropyLoss(weight=train_weights)

total_batches = len(train_loader)
report_interval = max(1, total_batches // 5) # report every 20% batch processed
print("total batches", total_batches, "| report_interval", report_interval)

min_epoch = 0
last_epoch=0
history_df = pd.DataFrame()
best_val_loss=1e20

VERSION = 1

load_state = True
if (load_state):
    history_df = pd.read_csv(f"../GNN_files/GNN_{mode}_{extension}_train_history_v{VERSION}_all.csv")
    print("Loading Model State from file!")
    print(history_df)
    
    last_epoch = int(max(history_df.epoch)) # Carry on from the last epoch
    checkpoint = torch.load(f'../GNN_files/GNN_{mode}_{extension}_v{VERSION}_all_epoch{last_epoch}.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    min_epoch     = checkpoint['epoch']+1
    best_val_loss = checkpoint['loss']

# ---------------------------------------------------------------------------------------------------
EPOCHS = 2


for epoch in range(min_epoch, min_epoch+EPOCHS):
    torch_nvtx.range_push(f"Epoch_{epoch}")
    
    
    print("Running Epoch ", epoch)
    t0 = time.time()

    # -------- training --------
    # Put the model in training mode, gradients will be computed
    model.train() 
    train_loss = 0.0 # Sum the loss over all batches, then average it

    # Loop over training batches
    for i, data in enumerate(train_loader):
        torch_nvtx.range_push(f"Batch_{i}")
        data = data.to(device)
        optimizer.zero_grad()            # Clear old features
        
        torch_nvtx.range_push("Forward_Pass")
        logits = model(data)             # Data flows through the network, the output is called logits (raw scores not probabilities) -- forward pass
        loss = criterion(logits, data.y) # Compares the predictions (logits) with the true labels (yb)
        torch_nvtx.range_pop()
        
        torch_nvtx.range_push("Backward_Pass")
        loss.backward()                  # Computes the gradients of the loss (stored in param.grad for every model param)
        optimizer.step()                 # Uses the gradients to update the weights. Depends on the optimizer e.g. SGD, Adam etc -- this is the learning step
        train_loss += loss.item() * data.num_graphs # Accumulates the total loss. loss.item() returns the loss for this batch. So multiply by batch size data.num_graphs 
        torch_nvtx.range_pop()

        # Print every 10%
        if (i + 1) % report_interval == 0:
            percent_done = (i + 1) / total_batches * 100
            print(f"Progress Epoch {epoch}: {percent_done:.0f}% ({i + 1}/{total_batches} batches)")
        
        torch_nvtx.range_pop()
        if (i > 100):
            torch_nvtx.range_pop()
            break
        
        

    train_loss /= len(train_loader.dataset) # Average the training loss over the total dataset
