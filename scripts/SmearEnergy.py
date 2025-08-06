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


# Load in the hits
print("Loading hits")
print("Filename: ",  sys.argv[1]+".h5")
hits   = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
parts  = pd.read_hdf(sys.argv[1]+".h5", 'MC/particles')
config = pd.read_hdf(sys.argv[1]+".h5", 'MC/configuration')
print("Finished loading hits")

# cube_size = config[config["param_key"] == "/Geometry/ATPC/cube_size"].param_value.iloc[0]
# match = re.search(r'\d+\.\d+|\d+', cube_size)
# cube_size = float(match.group())*1000
cube_size=2600
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

def FilterEventE(hits, parts_, Emin, Emax, E_mean):

    event_energies = hits.groupby("event_id").n.sum()*E_mean*1000

    passed_events = event_energies[ (event_energies >= Emin) & ((event_energies <= Emax))].index

    hits_filtered  = hits[hits["event_id"].isin(passed_events)]
    parts_filtered = parts_[parts_["event_id"].isin(passed_events)]

    return hits_filtered, parts_filtered

# Returns true if the event is fully contained
def CheckHitBounds(df, R, z_min, z_max):
    outside = (df.x**2 + df.y**2 > R**2) | (df.z < z_min) | (df.z > z_max)
    return not outside.any()

dfsE1 = []
dfsE2 = []

for index, e in enumerate(hits.event_id.unique()):
    # print("On Event:", e )

    # Select the event
    eventE1 = hits[hits.event_id == e]
    partE1 = parts[parts.event_id == e]

    # Calc number of electrons in a hit
    eventE1["n"] = round(eventE1["energy"]/E_mean)
    eventE2 = eventE1.copy()
    partE2 = partE1.copy()
    
    # Smear the energy by different amounts
    eventE1["n"]  = eventE1["n"].apply(lambda x: smear_energy(x, resolution=1))
    eventE2["n"]  = eventE2["n"].apply(lambda x: smear_energy(x, resolution=0.5))

    # Get the particle that deposited the most energy in the active, treat as primary
    eventE1_grouped = eventE1.groupby(["event_id", "particle_id"]).energy.sum()
    primary_part_id_E1 = eventE1_grouped.idxmax()[1] # Get particle_id with max energy from hits
    electronE1_creator = partE1[partE1.particle_id == primary_part_id_E1].creator_proc.iloc[0]
    eventE1["creator_proc"] = electronE1_creator
    eventE1["contained"] = CheckHitBounds(eventE1, cube_size/2.0-20, -cube_size/2.0+20, cube_size/2.0-20)

    eventE2_grouped = eventE2.groupby(["event_id", "particle_id"]).energy.sum()
    primary_part_id_E2 = eventE2_grouped.idxmax()[1] # Get particle_id with max energy from hits
    electronE2_creator = partE2[partE2.particle_id == primary_part_id_E2].creator_proc.iloc[0]
    eventE2["creator_proc"] = electronE2_creator
    eventE2["contained"] = CheckHitBounds(eventE2, cube_size/2.0-20, -cube_size/2.0+20, cube_size/2.0-20)
    
    dfsE1.append(eventE1)
    dfsE2.append(eventE2)

dfsE1 = pd.concat(dfsE1)
dfsE2 = pd.concat(dfsE2)

parts2 = parts.copy()

# Apply cut so that we only keep events within a certain energy window
dfsE1, parts  = FilterEventE(dfsE1, parts, 2433.3804, 2482.5396, E_mean)
dfsE2, _      = FilterEventE(dfsE2, parts2, 2445.6702, 2470.2498, E_mean)


N_gen   = config[config["param_key"] == "num_events"].param_value.iloc[0]
N_saved = config[config["param_key"] == "saved_events"].param_value.iloc[0]
N_savedE1 = len(dfsE1.event_id.unique())
N_savedE2 = len(dfsE2.event_id.unique())
N_savedE1G = len(dfsE1[dfsE1.creator_proc != "RadioactiveDecay"].event_id.unique()) # should be gamma num only
N_savedE2G = len(dfsE2[dfsE2.creator_proc != "RadioactiveDecay"].event_id.unique()) # should be gamma num only
N_savedE1C = len(dfsE1[dfsE1.contained == True].event_id.unique()) # apply 2 cm containment
N_savedE2C = len(dfsE2[dfsE2.contained == True].event_id.unique()) # apply 2 cm containment
min_E   = config[config["param_key"] == "/Actions/DefaultEventAction/min_energy"].param_value.iloc[0]
max_E   = config[config["param_key"] == "/Actions/DefaultEventAction/max_energy"].param_value.iloc[0]
# P       = config[config["param_key"] == "/Geometry/ATPC/gas_pressure"].param_value.iloc[0]
P       = 15
# detsize = config[config["param_key"] == "/Geometry/ATPC/cube_size"].param_value.iloc[0]
detsize=2.600
# chamber_thick = config[config["param_key"] == "/Geometry/ATPC/chamber_thickn"].param_value.iloc[0]
chamber_thick = 12
seed = config[config["param_key"] == "/nexus/random_seed"].param_value.iloc[0]
start_id = config[config["param_key"] == "/nexus/persistency/start_id"].param_value.iloc[0]

df_meta = pd.DataFrame({
"N_gen"     : [N_gen],
"N_saved"   : [N_saved],
"N_savedE1" : [N_savedE1],
"N_savedE2" : [N_savedE2],
"N_savedE1G" : [N_savedE1G],
"N_savedE2G" : [N_savedE2G],
"N_savedE1C" : [N_savedE1C],
"N_savedE2C" : [N_savedE2C],
"min_E"     : [min_E],
"max_E"     : [max_E],
"P"         : [P],
"detsize"   : [detsize],
"chamber_thick" : [chamber_thick],
"seed"      : [seed],
"start_id"  : [start_id]
})

# dfsE1 = dfsE1[["event_id", "x", "y", "z", "energy", "particle_id", "n"]]
dfsE2 = dfsE2[["event_id", "x", "y", "z", "energy", "particle_id", "n"]]

print(df_meta, "\n")
print(dfsE2)

# Write to the file
outfile = sys.argv[1] + "_Efilt.h5"

print("Saving events to file: ", outfile)
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/particles', parts, format='table')
    store.put('MC/hits', dfsE2, format='table')
    store.put('MC/configuration', config, format='table')
    store.put("MC/meta", df_meta, format='table')



