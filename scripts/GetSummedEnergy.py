import sys
import numpy  as np
import pandas as pd
import time

pd.options.mode.chained_assignment = None  # Disable the warning

# USAGE:
# python3 SmearEvents.py <name of nexus input file name (remove .h5 extension)> <JOBID>
# e.g. python3 SmearEvents.py /Users/mistryk2/Packages/nexus/ATPC_0nuBB 1

# Record the start time
start_time = time.time()

# Load in the hits
print("Filename: ", sys.argv[1]+".h5")
hits = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
print("Finished loading hits")

jobid = int(sys.argv[2])



# Get the hit tables and plot the total energy deposited in the TPC
# Sum the energy of the hits and bin into a histogram
hit_tot_energy = hits.groupby('event_id')['energy'].sum().reset_index(name='hit_tot_energy')

print(hit_tot_energy)


outfile = sys.argv[1] + "_"+ str(jobid) + ".h5"

print("Saving events to file: ", outfile)
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    store.put('MC/E', hit_tot_energy, format='table')

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")