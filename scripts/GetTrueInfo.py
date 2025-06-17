import pandas as pd
import numpy as np
import glob
import sys
import os

# Function gets the energy based on a sphere of radius radius_threshold
def GetBlobEnergyRadius(parts_elec, hits_all, end, radius_threshold):
    
    # Get coordinates if the start/end
    if (end == "start"):
        start_coord = parts_elec[['initial_x', 'initial_y', 'initial_z']].values
    else:
        start_coord = parts_elec[['final_x', 'final_y', 'final_z']].values

    # Calculate the Euclidean distance from each row
    distances = np.sqrt(((hits_all[['x', 'y', 'z']].values[:, None] - start_coord) ** 2).sum(axis=2))

    # Find rows where any distance is within the threshold, then sum their energy
    mask = (distances < radius_threshold).any(axis=1)
    result = hits_all[mask]
    return result.energy.sum()


def SortBlobs(blob1_E, blob2_E):

    if (blob1_E > blob2_E):
        return blob1_E, blob2_E
    else:
        return blob2_E, blob1_E


# Get the primary track kinetic energy and all the daughters if they are electrons
def GetPrimaryKE(parts, hits, particle_id, delta_thresh):

    # Get the energy of the parent particle
    tot_E = hits[hits.particle_id == particle_id].energy.sum()

    # Get all the daughters of the particle
    daughter_pids = parts[ (parts.mother_id == particle_id) & (parts.kin_energy <= delta_thresh) & (parts.particle_name == "e-")].particle_id.unique()

    # Loop over daughters
    for daughter in daughter_pids:

        # Get any particles who have a mother that is the daughter and is in threshold
        daughter_part = parts[ (parts.mother_id == daughter) & (parts.kin_energy <= delta_thresh) & (parts.particle_name == "e-")].particle_id.unique()

        # If there are daughers to the daughter then cycle
        if (len(daughter_part) > 0):
            daughter_E = GetPrimaryKE(parts, hits, daughter, delta_thresh)
        # Otherwise we can add the energy as is
        else:
            daughter_E = hits[hits.particle_id == daughter].energy.sum()
        
        tot_E = tot_E+daughter_E

    return tot_E



def CalcTrackExtent(hits):

    diff = hits.groupby('event_id').agg({
        'x': lambda x: x.max() - x.min(),
        'y': lambda y: y.max() - y.min(),
        'z': lambda z: z.max() - z.min()
    }).reset_index()

    # Rename columns
    diff.rename(columns={'x': 'dx', 'y': 'dy', 'z': 'dz'}, inplace=True)

    diff["Diam"] = np.sqrt((diff.dx)**2 + (diff.dy)**2 + (diff.dz)**2)

    return diff["Diam"]



def GetTrueInfoSignal(parts, hits, pressure):

    lengths  = []
    energies = []
    energies1 = [] # 0.1 MeV Delta thresh
    energies2 = [] # 0.2 MeV Delta thresh
    energies3 = [] # 0.5 MeV Delta thresh
    blob1_Es = []
    blob2_Es = []
    creator_procs = []
    event_ids = []

    for eid in parts.event_id.unique():

        # print("\n\n On event:", eid)

        part_event = parts[parts.event_id == eid]
        hits_event = hits[hits.event_id == eid]
        electron1  = part_event[part_event.particle_id == 1]
        electron2  = part_event[part_event.particle_id == 2]

        electron1_E = hits_event[hits_event.particle_id == 1].energy.sum()
        electron2_E = hits_event[hits_event.particle_id == 2].energy.sum()

        length = electron1.length.iloc[0]     + electron2.length.iloc[0] # total length

        tot_KE  = GetPrimaryKE(part_event, hits_event, 1, 2.6) +  GetPrimaryKE(part_event, hits_event, 2, 2.6)
        tot_KE1 = GetPrimaryKE(part_event, hits_event, 1, 0.1) +  GetPrimaryKE(part_event, hits_event, 2, 0.1)
        tot_KE2 = GetPrimaryKE(part_event, hits_event, 1, 0.2) +  GetPrimaryKE(part_event, hits_event, 2, 0.2)
        tot_KE3 = GetPrimaryKE(part_event, hits_event, 1, 0.5) +  GetPrimaryKE(part_event, hits_event, 2, 0.5)
        # tot_KE = electron1_E + electron2_E # total energy

        blob1_E =  GetBlobEnergyRadius(electron1, hits_event, "end", 180/pressure)
        blob2_E =  GetBlobEnergyRadius(electron2, hits_event, "end",   180/pressure)
        blob1_E, blob2_E = SortBlobs(blob1_E, blob2_E) # Make sure the blohE are labelled properly

        # print("Length:", length, "mm")
        # print("Tot Energy:", tot_KE, "MeV")
        # print("Blob1 Energy:", blob1_E, "MeV")
        # print("Blob2 Energy:", blob2_E, "MeV")

        lengths.append(length)
        energies.append(tot_KE)
        energies1.append(tot_KE1)
        energies2.append(tot_KE2)
        energies3.append(tot_KE3)
        blob1_Es.append(blob1_E)
        blob2_Es.append(blob2_E)
        creator_procs.append("DBD")
        event_ids.append(eid)

    TrackDiam = CalcTrackExtent(hits[ (hits.particle_id == 1) | (hits.particle_id == 2) ] )

    return pd.DataFrame({ "event_id": event_ids, "TrackLength" : lengths, "TrackEnergy" : energies, "TrackEnergy1" : energies1,  "TrackEnergy2" : energies2, "TrackEnergy3" : energies3,  "Blob1E" : blob1_Es, "Blob2E" :blob2_Es, "TrackDiam" :TrackDiam, "CreatorProc" :creator_procs})


def GetTrueInfoSingle(parts, hits, pressure):

    lengths  = []
    energies = []
    energies1 = [] # 0.1 MeV Delta thresh
    energies2 = [] # 0.2 MeV Delta thresh
    energies3 = [] # 0.5 MeV Delta thresh
    blob1_Es = []
    blob2_Es = []
    creator_procs = []
    event_ids = []

    for eid in parts.event_id.unique():

        # print("\n\n On event:", eid)

        part_event = parts[parts.event_id == eid]
        hits_event = hits[hits.event_id == eid]
        electron1  = part_event[part_event.particle_id == 1]

        electron1_E = hits_event[hits_event.particle_id == 1].energy.sum()

        length = electron1.length.iloc[0]

        tot_KE  = GetPrimaryKE(part_event, hits_event, 1, 2.6)
        tot_KE1 = GetPrimaryKE(part_event, hits_event, 1, 0.1)
        tot_KE2 = GetPrimaryKE(part_event, hits_event, 1, 0.2)
        tot_KE3 = GetPrimaryKE(part_event, hits_event, 1, 0.5)
        # tot_KE = electron1_E + electron2_E # total energy

        blob1_E =  GetBlobEnergyRadius(electron1, hits_event, "end", 180/pressure)
        blob2_E =  GetBlobEnergyRadius(electron1, hits_event, "start",   180/pressure)

        # print("Length:", length, "mm")
        # print("Tot Energy:", tot_KE, "MeV")
        # print("Blob1 Energy:", blob1_E, "MeV")
        # print("Blob2 Energy:", blob2_E, "MeV")

        lengths.append(length)
        energies.append(tot_KE)
        energies1.append(tot_KE1)
        energies2.append(tot_KE2)
        energies3.append(tot_KE3)
        blob1_Es.append(blob1_E)
        blob2_Es.append(blob2_E)
        creator_procs.append("single")
        event_ids.append(eid)

    TrackDiam = CalcTrackExtent(hits[ (hits.particle_id == 1)] )

    return pd.DataFrame({ "event_id": event_ids, "TrackLength" : lengths, "TrackEnergy" : energies, "TrackEnergy1" : energies1,  "TrackEnergy2" : energies2, "TrackEnergy3" : energies3,  "Blob1E" : blob1_Es, "Blob2E" :blob2_Es, "TrackDiam" :TrackDiam, "CreatorProc" :creator_procs})


def GetTrueInfoBackground(parts, hits, pressure):

    lengths  = []
    energies = []
    energies1 = [] # 0.1 MeV Delta thresh
    energies2 = [] # 0.2 MeV Delta thresh
    energies3 = [] # 0.5 MeV Delta thresh
    blob1_Es = []
    blob2_Es = []
    creator_procs = []
    event_ids = []
    TrackDiams = []


    for eid in parts.event_id.unique():

        # print("\n\n On event:", eid)

        part_event = parts[parts.event_id == eid]
        hits_event = hits[hits.event_id == eid]

        # Get the particle that deposited the most energy in the active, treat as primary
        hits_grouped = hits_event.groupby(["event_id", "particle_id"]).energy.sum()
        primary_part_id = hits_grouped.idxmax()[1] # Get particle_id with max energy from hits
        
        electron1  = part_event[part_event.particle_id == primary_part_id]
        creator_proc = electron1.creator_proc.iloc[0]


        tot_KE  = GetPrimaryKE(part_event, hits_event, primary_part_id, 2.6)
        tot_KE1 = GetPrimaryKE(part_event, hits_event, primary_part_id, 0.1)
        tot_KE2 = GetPrimaryKE(part_event, hits_event, primary_part_id, 0.2)
        tot_KE3 = GetPrimaryKE(part_event, hits_event, primary_part_id, 0.5)
        # tot_KE = hits_event[hits_event.particle_id == primary_part_id].energy.sum()

        length = electron1.length.iloc[0] # total length

        blob1_E =  GetBlobEnergyRadius(electron1, hits_event, "start", 180/pressure)
        blob2_E =  GetBlobEnergyRadius(electron1, hits_event, "end",   180/pressure)
        blob1_E, blob2_E = SortBlobs(blob1_E, blob2_E) # Make sure the blohE are labelled properly

        # print("Length:", length, "mm")
        # print("Tot Energy:", tot_KE, "MeV")
        # print("Blob1 Energy:", blob1_E, "MeV")
        # print("Blob2 Energy:", blob2_E, "MeV")

        lengths.append(length)
        energies.append(tot_KE)
        energies1.append(tot_KE1)
        energies2.append(tot_KE2)
        energies3.append(tot_KE3)
        blob1_Es.append(blob1_E)
        blob2_Es.append(blob2_E)
        creator_procs.append(creator_proc)
        event_ids.append(eid)

        TrackDiams.append(CalcTrackExtent(hits_event[ hits_event.particle_id == primary_part_id]).iloc[0])

    return pd.DataFrame({ "event_id": event_ids, "TrackLength" : lengths, "TrackEnergy" : energies, "TrackEnergy1" : energies1,  "TrackEnergy2" : energies2, "TrackEnergy3" : energies3, "Blob1E" : blob1_Es, "Blob2E" :blob2_Es, "TrackDiam" :TrackDiams, "CreatorProc" :creator_procs})



# load in the particles table

pressure = int(sys.argv[1])
mode     = sys.argv[2]
infile   = sys.argv[3]
jobid = int(sys.argv[4])
file_out_seg = os.path.basename(infile.rsplit('.', 1)[0])
file_out = f"{file_out_seg}_trueinfo_{jobid}.h5"

print("Mode:", mode)
print("infile:", infile)
print("output_file:", file_out)

dfs = []

print("Pressure:", pressure, "bar")

parts = pd.read_hdf(infile, "MC/particles")
hits  = pd.read_hdf(infile, "MC/hits")

if (mode == "0nubb"):
    df = GetTrueInfoSignal(parts, hits, pressure)
elif (mode == "single"):
    df = GetTrueInfoSingle(parts, hits, pressure)
else:
    df = GetTrueInfoBackground(parts, hits, pressure)

df["pressure"] = pressure

dfs.append(df)

dfs = pd.concat(dfs)

pd.set_option('display.max_rows', None)  # Show all rows
print(dfs)

with pd.HDFStore(f"{file_out}", mode='w', complevel=5, complib='zlib') as store:
    store.put('trueinfo', dfs, format='table')