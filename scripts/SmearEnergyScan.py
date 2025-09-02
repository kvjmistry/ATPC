import pandas as pd
import numpy as np
import sys
import re
pd.options.mode.chained_assignment = None  # Disable the warning

# This script loads in the nexus hits, applies an energy smearing to the hits according to 
# 1% energy and 0.5% resolution at Qbb. Then filters the events based on the Qbb ROI.
# This ensures we only smear events that land in the QBB region boosting the statistics
# We keep track of the number of generated events and number of saved events before E res
# cuts in a metadata table. Only the hits with 1% ROI window are kept

# To run
# python3 SmearEnergy.py <name without extension>

# Load in the hits
print("Loading hits")
print("Filename: ",  sys.argv[1]+".h5")
hits   = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
parts  = pd.read_hdf(sys.argv[1]+".h5", 'MC/particles')
config = pd.read_hdf(sys.argv[1]+".h5", 'MC/configuration')
print("Finished loading hits")

cube_size = config[config["param_key"] == "/Geometry/ATPC/cube_size"].param_value.iloc[0]
match = re.search(r'\d+\.\d+|\d+', cube_size)
cube_size = float(match.group())*1000
# cube_size=2600
print("The cube_size is:", cube_size)

# Mean energy per e-. This splits up each G4 into E_hit/E_mean electrons
E_mean = 24.8e-6 # [eV]

# Function to smear the number of electrons in each hit by the fano factor
def smear_energy(N, resolution):
    if N < 10:
        return np.random.poisson(N)  # Poisson for small N
    else:
        # sigma = np.sqrt(N * 0.15) # 0.15 Fano factor
        if (resolution == 1):
            sigma = np.sqrt(N * 1.8) # 1% ER
        
        elif (resolution == 1.2):
            sigma = np.sqrt(N * 2.6) # 1.2% ER
        
        elif (resolution == 0.3):
            sigma = np.sqrt(N * 0.15) # 0.3% ER
        
        elif (resolution == 0.75):
            sigma = np.sqrt(N * 0.98) # 0.75% ER
        
        elif (resolution == 0.5):
            sigma = np.sqrt(N * 0.45) # 0.5% ER
        else:
            print("Error resoltion not defined, using default 1%")
            sigma = np.sqrt(N * 1.8) # 1% ER
        
        new_n = int(round(np.random.normal(N, sigma)))
        if (new_n < 0):
            return 1
        else:
            return new_n  # Apply Gauss+rounding

def FilterEventE(hits, parts_, Eres, E_mean):

    min_E = 2458 - Eres*2458/100
    max_E = 2458 + Eres*2458/100

    event_energies = hits.groupby("event_id").n.sum()*E_mean*1000

    passed_events = event_energies[ (event_energies >= min_E) & ((event_energies <= max_E))].index

    hits_filtered  = hits[hits["event_id"].isin(passed_events)]
    parts_filtered = parts_[parts_["event_id"].isin(passed_events)]

    return hits_filtered, parts_filtered

# Returns true if the event is fully contained
def CheckHitBounds(df, R, z_min, z_max):
    outside = (df.x**2 + df.y**2 > R**2) | (df.z < z_min) | (df.z > z_max)
    return not outside.any()


def ApplySmearing(hits, parts, e, Eres):
    # Select the event
    event = hits[hits.event_id == e]
    part  = parts[parts.event_id == e]

    # Calc number of electrons in a hit
    event["n"] = round(event["energy"]/E_mean)
    
    # Smear the energy by different amounts
    event["n"]  = event["n"].apply(lambda x: smear_energy(x, resolution=Eres))

    # Get the particle that deposited the most energy in the active, treat as primary
    event_grouped         = event.groupby(["event_id", "particle_id"]).energy.sum()
    primary_part_id       = event_grouped.idxmax()[1] # Get particle_id with max energy from hits
    electron_creator      = part[part.particle_id == primary_part_id].creator_proc.iloc[0]
    event["creator_proc"] = electron_creator
    event["contained"]    = CheckHitBounds(event, cube_size/2.0-20, -cube_size/2.0+20, cube_size/2.0-20)

    return event

dfsE1 = []
dfsE2 = []
dfsE3 = []
dfsE4 = []
dfsE5 = []

print("Looping events, total events to process:", len(hits.event_id.unique()))
for index, e in enumerate(hits.event_id.unique()):
    # print("On Event:", e )

    dfsE1.append(ApplySmearing(hits, parts, e, 1.2))
    dfsE2.append(ApplySmearing(hits, parts, e, 1.0))
    dfsE3.append(ApplySmearing(hits, parts, e, 0.75))
    dfsE4.append(ApplySmearing(hits, parts, e, 0.5))
    dfsE5.append(ApplySmearing(hits, parts, e, 0.3))

dfsE1 = pd.concat(dfsE1)
dfsE2 = pd.concat(dfsE2)
dfsE3 = pd.concat(dfsE3)
dfsE4 = pd.concat(dfsE4)
dfsE5 = pd.concat(dfsE5)

parts1 = parts.copy()
parts2 = parts.copy()
parts3 = parts.copy()
parts4 = parts.copy()
parts5 = parts.copy()

print("Filtering event energies")
# Apply cut so that we only keep events within a certain energy window
dfsE1, parts1  = FilterEventE(dfsE1, parts, 1.2,  E_mean)
dfsE2, parts2  = FilterEventE(dfsE2, parts, 1.0,  E_mean)
dfsE3, parts3  = FilterEventE(dfsE3, parts, 0.75, E_mean)
dfsE4, parts4  = FilterEventE(dfsE4, parts, 0.5,  E_mean)
dfsE5, parts5  = FilterEventE(dfsE5, parts, 0.3,  E_mean)

N_gen   = config[config["param_key"] == "num_events"].param_value.iloc[0]
N_saved = config[config["param_key"] == "saved_events"].param_value.iloc[0]
N_savedE1 = len(dfsE1.event_id.unique())
N_savedE2 = len(dfsE2.event_id.unique())
N_savedE3 = len(dfsE3.event_id.unique())
N_savedE4 = len(dfsE4.event_id.unique())
N_savedE5 = len(dfsE5.event_id.unique())
min_E   = config[config["param_key"] == "/Actions/DefaultEventAction/min_energy"].param_value.iloc[0]
max_E   = config[config["param_key"] == "/Actions/DefaultEventAction/max_energy"].param_value.iloc[0]
P       = config[config["param_key"] == "/Geometry/ATPC/gas_pressure"].param_value.iloc[0]
detsize = config[config["param_key"] == "/Geometry/ATPC/cube_size"].param_value.iloc[0]
chamber_thick = config[config["param_key"] == "/Geometry/ATPC/chamber_thickn"].param_value.iloc[0]
seed = config[config["param_key"] == "/nexus/random_seed"].param_value.iloc[0]
start_id = config[config["param_key"] == "/nexus/persistency/start_id"].param_value.iloc[0]

df_meta = pd.DataFrame({
"N_gen"     : [N_gen],
"N_saved"   : [N_saved],
"N_savedE1" : [N_savedE1],
"N_savedE2" : [N_savedE2],
"N_savedE3" : [N_savedE3],
"N_savedE4" : [N_savedE4],
"N_savedE5" : [N_savedE5],
"min_E"     : [min_E],
"max_E"     : [max_E],
"P"         : [P],
"detsize"   : [detsize],
"chamber_thick" : [chamber_thick],
"seed"      : [seed],
"start_id"  : [start_id]
})

dfsE1 = dfsE1[["event_id", "x", "y", "z", "energy", "particle_id", "n"]]
dfsE2 = dfsE2[["event_id", "x", "y", "z", "energy", "particle_id", "n"]]
dfsE3 = dfsE3[["event_id", "x", "y", "z", "energy", "particle_id", "n"]]
dfsE4 = dfsE4[["event_id", "x", "y", "z", "energy", "particle_id", "n"]]
dfsE5 = dfsE5[["event_id", "x", "y", "z", "energy", "particle_id", "n"]]

print(df_meta, "\n")

# Write to the file
outfile = sys.argv[1] + "_Efilt.h5"

print("Saving events to file: ", outfile)
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/particles', parts5, format='table')
    store.put('MC/hits1', dfsE1, format='table')
    store.put('MC/hits2', dfsE2, format='table')
    store.put('MC/hits3', dfsE3, format='table')
    store.put('MC/hits4', dfsE4, format='table')
    store.put('MC/hits5', dfsE5, format='table')
    store.put('MC/configuration', config, format='table')
    store.put("MC/meta", df_meta, format='table')



