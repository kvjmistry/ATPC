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


def GetPrimaryKE(parts, hits, particle_id):

    primary_E = hits[hits.particle_id == particle_id].energy.sum()

    daugter_pids = parts[ (parts.mother_id == particle_id) & (parts.kin_energy < 0.1)].particle_id.unique()
    # display(parts)

    for daughter in daugter_pids:
        daugher_E = hits[hits.particle_id == daughter].energy.sum()
        primary_E = primary_E+daugher_E

    return primary_E


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

        tot_KE = GetPrimaryKE(part_event, hits_event, 1) +  GetPrimaryKE(part_event, hits_event, 2)
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
        blob1_Es.append(blob1_E)
        blob2_Es.append(blob2_E)
        creator_procs.append("DBD")
        event_ids.append(eid)

    TrackDiam = CalcTrackExtent(hits[ (hits.particle_id == 1) | (hits.particle_id == 2) ] )

    return pd.DataFrame({ "event_id": event_ids, "TrackLength" : lengths, "TrackEnergy" : energies, "Blob1E" : blob1_Es, "Blob2E" :blob2_Es, "TrackDiam" :TrackDiam, "CreatorProc" :creator_procs})


def GetTrueInfoBackground(parts, hits, pressure):

    lengths  = []
    energies = []
    blob1_Es = []
    blob2_Es = []
    creator_procs = []
    event_ids = []


    for eid in parts.event_id.unique():

        # print("\n\n On event:", eid)

        part_event = parts[parts.event_id == eid]

        # Here we get the row with the largest electron energy and call that the primary
        part_event = part_event[part_event.particle_name == "e-"]
        parts_primary = part_event.loc[[part_event['kin_energy'].idxmax()]]
        primary_part_id = parts_primary.particle_id.iloc[0]
        
        hits_event = hits[hits.event_id == eid]
        electron1  = part_event[part_event.particle_id == primary_part_id]
        creator_proc = electron1.creator_proc.iloc[0]


        tot_KE = GetPrimaryKE(part_event, hits_event, primary_part_id)
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
        blob1_Es.append(blob1_E)
        blob2_Es.append(blob2_E)
        creator_procs.append(creator_proc)
        event_ids.append(eid)

    TrackDiam = CalcTrackExtent(hits[ hits.particle_id == primary_part_id])

    return pd.DataFrame({ "event_id": event_ids, "TrackLength" : lengths, "TrackEnergy" : energies, "Blob1E" : blob1_Es, "Blob2E" :blob2_Es, "TrackDiam" :TrackDiam, "CreatorProc" :creator_procs})



# load in the particles table

pressure = sys.argv[1]
mode     = sys.argv[2]
infile   = sys.argv[3]
file_out_seg = os.path.basename(infile.rsplit('.', 1)[0])
file_out = f"{file_out_seg}_properties.h5"

print("Mode:", mode)
print("infile:", infile)
print("output_file:", file_out)

dfs = []

print("Pressure:", pressure, "bar")

parts = pd.read_hdf(infile, "MC/particles")
hits  = pd.read_hdf(infile, "MC/hits")

if (mode == "0nubb"):
    df = GetTrueInfoSignal(parts, hits, pressure)
else:
    df = GetTrueInfoBackground(parts, hits, pressure)

df["pressure"] = pressure

dfs.append(df)

dfs = pd.concat(dfs)

with pd.HDFStore(f"{file_out}", mode='w', complevel=5, complib='zlib') as store:
    store.put('trueinfo', dfs, format='table')