import pandas as pd
import numpy as np
import sys
import re
pd.options.mode.chained_assignment = None  # Disable the warning

# This script loads in the nexus hits, applies an energy smearing to the hits according to 
# 0.5% resolution at Qbb. Then filters the events based on a window of 2.4 - 2.5 MeV.
# This ensures we only smear events that land in the QBB region boosting the statistics
# We also add a flag to see if the event is contained or not


# Load in the hits
print("Loading hits")
print("Filename: ",  sys.argv[1]+".h5")
hits   = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
parts  = pd.read_hdf(sys.argv[1]+".h5", 'MC/particles')
config = pd.read_hdf(sys.argv[1]+".h5", 'MC/configuration')
print("Finished loading hits")

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
def CheckHitBounds(df):
    R = cube_size/2.0-20
    z1_max = cube_size/2.0-20
    z1_min = 20.1 # mm
    z2_max = -20.1 #mm
    z2_min = -cube_size/2.0+20
    
    outside = (df.x**2 + df.y**2 > R**2) | (df.z < z2_min) | (df.z > z1_max) | ((df.z > z2_max) & (df.z < z1_min))
    return not outside.any()

dfs = []

for index, e in enumerate(hits.event_id.unique()):
    print("On Event:", e )

    # Select the event
    event = hits[hits.event_id == e]
    part  = parts[parts.event_id == e]

    # Calc number of electrons in a hit
    event["n"] = round(event["energy"]/E_mean)
    
    # Smear the energy by different amounts
    event["n"]  = event["n"].apply(lambda x: smear_energy(x, resolution=0.5))

    # Get the particle that deposited the most energy in the active, treat as primary
    event_grouped = event.groupby(["event_id", "particle_id"]).energy.sum()
    primary_part_id = event_grouped.idxmax()[1] # Get particle_id with max energy from hits
    electron_creator = part[part.particle_id == primary_part_id].creator_proc.iloc[0]
    event["creator_proc"] = electron_creator
    event["contained"] = CheckHitBounds(event)
    dfs.append(event)

dfs = pd.concat(dfs)

# Apply cut so that we only keep events within a certain energy window
dfs, parts  = FilterEventE(dfs, parts, 2400, 2500, E_mean)


N_gen      = config[config["param_key"] == "num_events"].param_value.iloc[0]
N_saved    = config[config["param_key"] == "saved_events"].param_value.iloc[0]
N_savedWin = len(dfs.event_id.unique())
N_savedG   = len(dfs[dfs.creator_proc != "RadioactiveDecay"].event_id.unique()) # should be gamma num only
N_savedC   = len(dfs[dfs.contained == True].event_id.unique()) # apply 2 cm containment
min_E      = config[config["param_key"] == "/Actions/DefaultEventAction/min_energy"].param_value.iloc[0]
max_E      = config[config["param_key"] == "/Actions/DefaultEventAction/max_energy"].param_value.iloc[0]
P          = 15
detsize    = 2.600
chamber_thick = 12
seed     = config[config["param_key"] == "/nexus/random_seed"].param_value.iloc[0]
start_id = config[config["param_key"] == "/nexus/persistency/start_id"].param_value.iloc[0]

df_meta = pd.DataFrame({
"N_gen"      : [N_gen],
"N_saved"    : [N_saved],
"N_savedWin" : [N_savedWin],
"N_savedG"   : [N_savedG],
"N_savedC"   : [N_savedC],
"min_E"      : [min_E],
"max_E"      : [max_E],
"P"          : [P],
"detsize"    : [detsize],
"chamber_thick" : [chamber_thick],
"seed"       : [seed],
"start_id"   : [start_id]
})

# Reset the energy
dfs["energy"] = dfs["n"]*E_mean

dfs = dfs[["event_id", "x", "y", "z", "energy", "particle_id"]]

# Here we voxelize the hits

# Min x val, max x val, x bin w (z and y are set equal to this)
xmin = -cube_size/2.0
xmax = cube_size/2.0
xbw  = 3

xbins = np.arange(xmin, xmax+xbw, xbw)
ybins = xbins
zbins = xbins
xbins_centre = np.arange(xmin+xbw/2, xmax+xbw/2, xbw)
ybins_centre = xbins_centre
zbins_centre = xbins_centre

# Now bin the x, y, z positions
dfs['x'] = pd.cut(x=dfs['x'], bins=xbins,labels=xbins_centre, include_lowest=True)
dfs['y'] = pd.cut(x=dfs['y'], bins=ybins,labels=ybins_centre, include_lowest=True)
dfs['z'] = pd.cut(x=dfs['z'], bins=zbins,labels=zbins_centre, include_lowest=True)

dfs['x'] = dfs['x'].astype(float)
dfs['y'] = dfs['y'].astype(float)
dfs['z'] = dfs['z'].astype(float)
dfs['z'] = dfs['z']+1300

print(df_meta, "\n")
print(dfs)

# Write to the file
outfile = sys.argv[1] + "_Efilt.h5"

print("Saving events to file: ", outfile)
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/particles', parts, format='table')
    store.put('MC/hits', dfs, format='table')
    store.put('MC/configuration', config, format='table')
    store.put("MC/meta", df_meta, format='table')



