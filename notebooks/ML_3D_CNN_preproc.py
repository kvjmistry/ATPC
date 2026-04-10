# This notebook is for training the ATPC data with a 3D CNN
# Loads in preprocessed image files that have been pre-shuffled and voxelized

import pandas as pd
import numpy as np
import os
import glob
import json
import time
from tqdm import tqdm
import lmdb
import pickle
import torch.cuda.nvtx as torch_nvtx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

import spconv.pytorch as spconv

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


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

# SpCov expects batch index to be ordered from 0 to batch_size -1
# When we run in batches, we get random indexes
# to fix this during training we shift the set so the batch index goes back from 0 to 1 
# i.e. we are using local batch index during training
def spconv_collate_fn(batch):
    coords_list, feats_list, labels_list, meta_list = zip(*batch)
    
    # Update the batch index (first column of coords)
    # Without this, spconv thinks every point in the batch belongs to 'Event 0'
    new_coords_list = []
    for i, coords in enumerate(coords_list):
        # Set the first column to the actual index in this specific batch
        coords[:, 0] = i 
        new_coords_list.append(coords)
        
    # Concatenate everything into single large tensors
    batch_coords = torch.cat(new_coords_list, dim=0)
    batch_feats  = torch.cat(feats_list, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)
    
    # Structure meta list like 
    #{"event_id": [101, 102, 103, 104],"subType": ["TypeA", "TypeB", "TypeA", "TypeC"]}
    batch_meta = {
        key: [m[key] for m in meta_list] 
        for key in meta_list[0].keys()
    }
    
    return batch_coords, batch_feats, batch_labels, batch_meta

class LMDBEventDataset(Dataset):
    def __init__(self, lmdb_path, data_sample):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        
        # Load in the metadata
        # meta_file = f"/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//CNN_files/CNN_{data_sample}_meta.json"
        meta_file = f"CNN_{data_sample}_meta_test.json"
        
        print(f"Loading metadata from cache: {meta_file}")
        with open(meta_file, 'r') as f:
            cache_data = json.load(f)
            raw_counts = cache_data['subtype_counts']
            self.subtype_counts = Counter({int(k): v for k, v in raw_counts.items()})
            self.input_data_shape = cache_data["input_data_shape"]
            
        self.print_scale_factors(data_sample)
        print("The input shape is: ")
        print(self.input_data_shape)
        
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f"{idx:09d}".encode()
            data = pickle.loads(txn.get(key))
            
        return (
            torch.from_numpy(data["coords"]).int(),
            torch.from_numpy(data["energy"]).float(),
            torch.tensor(data["label"]),
            {"subType": data["subtype"], "event_id": data["event_id"]}
        )
        
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

# train_dataset = LMDBEventDataset("/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//CNN_files/CNN_Train_events.lmdb", "Train")
# val_dataset   = LMDBEventDataset("/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//CNN_files/CNN_Validation_events.lmdb", "Validation")
# test_dataset  = LMDBEventDataset("/media/argon/HardDrive_8TB/Krishan/ATPC/ML_samples//CNN_files/CNN_Test_events.lmdb", "Test")

train_dataset = LMDBEventDataset("CNN_Train_events_test.lmdb", "Train")
val_dataset   = LMDBEventDataset("CNN_Validation_events_test.lmdb", "Validation")
test_dataset  = LMDBEventDataset("CNN_Test_events_test.lmdb", "Test")


BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=spconv_collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=spconv_collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True, persistent_workers=True, prefetch_factor=2, collate_fn=spconv_collate_fn)



class SparseEventNet(nn.Module):
    def __init__(self, input_channels, num_classes, spatial_shape):
        super(SparseEventNet, self).__init__()
        self.spatial_shape = spatial_shape
        
        self.net = spconv.SparseSequential(
            # Block 1
            spconv.SubMConv3d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Block 2 
            spconv.SubMConv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Block 3: Downsample (Stride 2)
            spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 4
            spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Block 5: Final Downsample to a small feature map
            spconv.SparseConv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Final Dense Layer to cast to number of classes
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, features, coords, batch_size):
        
        x = spconv.SparseConvTensor(features, coords, self.spatial_shape, batch_size)
        x = self.net(x)
        out = self.global_pool(x)
        
        return self.fc(out)

    # Global Pooling
    # Take the features of the remaining sparse points and average+get max (per image)
    # x.features is a tensor of [Total_Active_Points, features]
    # We need to average these per-image using the batch indices in x.indices
    def global_pool(self, x):
        features = x.features
        batch_ids = x.indices[:, 0]
        pooled_feats = []
        for i in range(x.batch_size):
            mask = (batch_ids == i)
            if mask.any():
                pooled_feats.append(torch.cat([
                    features[mask].mean(dim=0),
                    features[mask].max(dim=0)[0]
                ]))
            # Safetey block in case a processed image is empty at this stage (no active voxels)
            # In this case return array of zeros. 
            else:
                pooled_feats.append(torch.zeros(features.shape[1]).to(features.device))
        return torch.stack(pooled_feats)


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
weight_bkg = total / (2 * total_bkg)
weight_signal = total / (2 * sig)

print("Weight signal:", weight_signal)
print("Weight Background:", weight_bkg)

# Create the model
model = SparseEventNet(input_channels=1, num_classes=2, spatial_shape=train_dataset.input_data_shape).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5) #  This will help adjust the learning rate
class_weights = torch.tensor([weight_bkg, weight_signal], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device) # can add this so it relaxes the loss criterion a bit. label_smoothing=0.1

total_batches = len(train_loader)
report_interval = max(1, total_batches // 5) # report every 20% batch processed
print("total batches", total_batches, "| report_interval", report_interval)

min_epoch = 0
last_epoch=0
history_df = pd.DataFrame()
best_val_loss=1e20

VERSION = 3 # This is with fix to the voxel function

load_state = True
if (load_state):
    history_df = pd.read_csv(f"../CNN_files/CNN_train_history_v{VERSION}.csv")
    print("Loading Model State from file!")
    
    last_epoch = int(max(history_df.epoch)) # Carry on from the last epoch
    checkpoint = torch.load(f'../CNN_files/CNN_v{VERSION}_epoch{last_epoch}.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    min_epoch     = checkpoint['epoch']+1
    best_val_loss = checkpoint['loss']

# ---------------------------------------------------------------------------------------------------
EPOCHS = 1

for epoch in range(min_epoch, min_epoch+EPOCHS):
    torch_nvtx.range_push(f"Epoch_{epoch}")
    
    print("Running Epoch ", epoch)
    t0 = time.time()

    # ---- train ---------------------------------------------------------------------------------
    model.train()
    train_loss = 0
    i = 0
    for i, (coords, feats, y, meta) in enumerate(train_loader):
        print(f"Batch_{i}")
        torch_nvtx.range_push(f"Batch_{i}")
        
        coords, feats, y = coords.to(device), feats.to(device), y.to(device)
        optimizer.zero_grad()
        
        torch_nvtx.range_push("Forward_Pass")
        logits = model(feats, coords, y.shape[0]) # array (Nbatch, [logit_bkg, logit_sig])
        loss = criterion(logits, y)
        torch_nvtx.range_pop()
        
        torch_nvtx.range_push("Backward_Pass")
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        torch_nvtx.range_pop()
        
        torch_nvtx.range_pop()

        # if (i > 505):
        #     torch_nvtx.range_pop()
        #     break
        
print("Done!")
