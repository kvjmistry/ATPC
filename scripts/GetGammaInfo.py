import sys
import numpy  as np
import pandas as pd
import time
import re

pd.options.mode.chained_assignment = None  # Disable the warning

def GetElecronGammaPair(part_event, index):

    gamma_df     = part_event[part_event.particle_name == "gamma"]
    # print(gamma_df)
    sorted_gamma = gamma_df.sort_values(by="kin_energy", ascending=False)
    # print("sorted gamma", len(sorted_gamma), index)
    # print(sorted_gamma)

    if index >= len(sorted_gamma):
        print(f"Warning: No gamma with associated electron found for event.", index)
        # print(gamma_df)
        # return None, None

    prim_gamma   = pd.DataFrame([sorted_gamma.iloc[index]])
    electron     = part_event[part_event.mother_id == prim_gamma.particle_id.iloc[0]]

    if (len(electron) == 0):
        prim_gamma, electron = GetElecronGammaPair(part_event, int(index+1))
    
    return prim_gamma, electron
    
def CheckHitBounds(df, R, z_min, z_max):
    outside = (df.x**2 + df.y**2 > R**2) | (df.z < z_min) | (df.z > z_max)
    return not outside.any()

def GetTrueInfoBackground(parts, hits, cube_size):

    lengths  = []
    energy = []
    creator_procs = []
    event_ids = []
    initial_x = []
    initial_y = []
    initial_z = []
    final_x = []
    final_y = []
    final_z = []
    energies = []
    mother_E=[]
    contained = []

    for eid in parts.event_id.unique():

        # print("\n\n On event:", eid)

        part_event = parts[parts.event_id == eid]
        hits_event = hits[hits.event_id == eid]

        # Get the particle that deposited the most energy in the active, treat as primary
        hits_grouped = hits_event.groupby(["event_id", "particle_id"]).energy.sum()
        primary_part_id = hits_grouped.idxmax()[1] # Get particle_id with max energy from hits

        electron = part_event[part_event.particle_id == primary_part_id]
        mother = part_event[part_event.particle_id == electron.mother_id.iloc[0]]

        contained.append(CheckHitBounds(hits_event, cube_size/2.0-20, -cube_size/2.0+20, cube_size/2.0-20))
        
        # If the electron came from a decay process, then use the electron info
        if (electron.creator_proc.iloc[0] == "RadioactiveDecay"):
            mother_E.append(electron.kin_energy.iloc[0])
            initial_x.append(round(electron.initial_x.iloc[0]))
            initial_y.append(round(electron.initial_y.iloc[0]))
            initial_z.append(round(electron.initial_z.iloc[0]))
        else:
            mother_E.append(mother.kin_energy.iloc[0])
            initial_x.append(round(mother.initial_x.iloc[0]))
            initial_y.append(round(mother.initial_y.iloc[0]))
            initial_z.append(round(mother.initial_z.iloc[0]))

        creator_procs.append(electron.creator_proc.iloc[0])
        final_x.append(round(electron.initial_x.iloc[0])) # This is the position of where the gamma interacted first
        final_y.append(round(electron.initial_y.iloc[0]))
        final_z.append(round(electron.initial_z.iloc[0]))
        
        
        energies.append(hits_event.groupby('event_id')['energy'].sum().item())

        event_ids.append(eid)

    return pd.DataFrame({ "event_id": event_ids, "CreatorProc" :creator_procs, "energy": energies, "mother_E": mother_E, "initial_x": initial_x, "initial_y": initial_y, "initial_z": initial_z,  "final_x": final_x, "final_y": final_y, "final_z": final_z, "contained": contained })

# USAGE:
# python3 SmearEvents.py <name of nexus input file name (remove .h5 extension)> <JOBID>
# e.g. python3 GetGammaInfo.py /Users/mistryk2/Packages/nexus/ATPC_0nuBB 1

# Record the start time
start_time = time.time()

# Load in the hits
print("Filename: ", sys.argv[1]+".h5")
hits = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
parts = pd.read_hdf(sys.argv[1]+".h5", 'MC/particles')
config = pd.read_hdf(sys.argv[1]+".h5", 'MC/configuration')
print("Finished loading hits and parts")

cube_size = config[config["param_key"] == "/Geometry/ATPC/cube_size"].param_value.iloc[0]
match = re.search(r'\d+\.\d+|\d+', cube_size)
cube_size = float(match.group())*1000
# cube_size=2600
print("The cube_size is:", cube_size)

jobid = int(sys.argv[2])

df = GetTrueInfoBackground(parts, hits, cube_size)
pd.set_option('display.max_rows', None)  # Show all rows
print(df)

outfile = sys.argv[1] + "_gammatable_" + str(jobid) + ".h5"

print("Saving events to file: ", outfile)
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    store.put('MC/E', df, format='table')

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")