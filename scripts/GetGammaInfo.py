import sys
import numpy  as np
import pandas as pd
import time

pd.options.mode.chained_assignment = None  # Disable the warning

def GetElecronGammaPair(part_event, index):

    gamma_df     = part_event[part_event.particle_name == "gamma"]
    sorted_gamma = gamma_df.sort_values(by="kin_energy", ascending=False)
    prim_gamma   = pd.DataFrame([sorted_gamma.iloc[index]])
    electron     = part_event[part_event.mother_id == prim_gamma.particle_id.iloc[0]]

    if (len(electron) == 0):
        prim_gamma, electron = GetElecronGammaPair(part_event, int(index+1))
    
    return prim_gamma, electron
    


def GetTrueInfoBackground(parts, hits):

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
    prim_gamma_E=[]

    for eid in parts.event_id.unique():

        # print("\n\n On event:", eid)

        part_event = parts[parts.event_id == eid]

        # Get the particle ID of the Bi/Tl gamma
        prim_gamma, electron = GetElecronGammaPair(part_event, 0)

        if (len(prim_gamma)  == 0):
            prim_gamma = pd.DataFrame([part_event.loc[part_event[part_event.particle_name == "gamma"].kin_energy.idxmax()]])

        prim_gamma_E.append(prim_gamma.kin_energy.iloc[0])
        initial_x.append(round(prim_gamma.initial_x.iloc[0]))
        initial_y.append(round(prim_gamma.initial_y.iloc[0]))
        initial_z.append(round(prim_gamma.initial_z.iloc[0]))

        # Sometimes the primary gamma escapes
        if (len(electron)  == 0):
                continue

        creator_procs.append(electron.creator_proc.iloc[0])
        final_x.append(round(electron.initial_x.iloc[0])) # This is the position of where the gamma interacted first
        final_y.append(round(electron.initial_y.iloc[0]))
        final_z.append(round(electron.initial_z.iloc[0]))
        
        
        hits_event = hits[hits.event_id == eid]
        energies.append(hits_event.groupby('event_id')['energy'].sum().item())

        event_ids.append(eid)

    return pd.DataFrame({ "event_id": event_ids, "CreatorProc" :creator_procs, "energy": energies, "prim_gamma_E": prim_gamma_E, "initial_x": initial_x, "initial_y": initial_y, "initial_z": initial_z,  "final_x": final_x, "final_y": final_y, "final_z": final_z })

# USAGE:
# python3 SmearEvents.py <name of nexus input file name (remove .h5 extension)> <JOBID>
# e.g. python3 GetGammaInfo.py /Users/mistryk2/Packages/nexus/ATPC_0nuBB 1

# Record the start time
start_time = time.time()

# Load in the hits
print("Filename: ", sys.argv[1]+".h5")
hits = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
print("Finished loading hits")

jobid = int(sys.argv[2])

df = GetTrueInfoBackground(parts, hits)
print(df)

outfile = sys.argv[1] + "_"+ str(jobid) + "_gammatable.h5"

print("Saving events to file: ", outfile)
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    store.put('MC/E', df, format='table')

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")