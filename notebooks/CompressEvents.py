# Script to read in the nexus files and smear the hits along the track.
# Re-bin the resultant track
import sys
import numpy  as np
import pandas as pd
from collections import Counter
import time

# Record the start time
start_time = time.time()

# Load in the hits
print("Loading hits")
print("Filename: ", sys.argv[1]+".h5")
hits = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
parts = pd.read_hdf(sys.argv[1]+".h5", 'MC/particles')
print("Finished loading hits")


# Filter the events to a set amount
# Get the first 60 unique events
event_list = hits['event_id'].unique()[0:60]

# Filter the DataFrame to keep only 50 events
hits = hits[hits['event_id'].isin(event_list)]
parts = parts[parts['event_id'].isin(event_list)]

print("Saving events to file: ", sys.argv[2]+".h5")
with pd.HDFStore(sys.argv[2]+".h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/particles', parts, format='table')
    store.put('MC/hits', hits, format='table')

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")