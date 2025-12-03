import pandas as pd
import numpy as np
import glob
import tables
import sys

# python MergeSophronia.py ldc1

def read_data(infile, run_number):

    try:
        reco    = pd.read_hdf(infile, key = 'RECO/Events')
    except (KeyError, OSError, IOError, tables.exceptions.HDF5ExtError):
        print(f"Missing data → removing file: {f}")
        #os.remove(f)
        return []

    dst     = pd.read_hdf(infile, key = 'DST/Events')
    runevts = pd.read_hdf(infile, key = 'Run/events')
    runinfo = pd.read_hdf(infile, key = 'Run/runInfo')

    if run_number <0:
        config        = pd.read_hdf(infile, key = 'MC/configuration')
        parts         = pd.read_hdf(infile, key = 'MC/particles')
        hits          = pd.read_hdf(infile, key = 'MC/hits')
        sns_response  = pd.read_hdf(infile, key = 'MC/sns_response')
        sns_positions = pd.read_hdf(infile, key = 'MC/sns_positions')
        evtmap        = pd.read_hdf(infile, key = 'Run/eventMap')

        return reco, dst, runevts, runinfo, config, parts, hits, sns_response, sns_positions, evtmap
    
    return reco, dst, runevts, runinfo
    
out_basename = sys.argv[1]
files = sorted(glob.glob(f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/230725/{out_basename}/*.h5"))
tot_files=len(files)
run_number=99

reco          = []
dst           = []
runevts       = []
runinfo       = []
config        = []
parts         = []
hits          = []
sns_response  = []
sns_positions = []
evtmap        = []

counter = 0

for index, f in enumerate(files):
    print("file index:",f"{index}/{tot_files}","name:",f)

    # MC file
    if (run_number < 0):
        reco_, dst_, runevts_, runinfo_, config_, parts_, hits_, sns_response_, sns_positions_, evtmap_ = read_data(f, run_number)
    # Data file
    else:
        reco_, dst_, runevts_, runinfo_ = read_data(f, run_number)

    if (len(reco_) == 0):
        continue

    reco.append(reco_)
    dst.append(dst_)
    runevts.append(runevts_)
    runinfo.append(runinfo_)
    
    if (run_number < 0):
        config.append(config_)
        parts.append(parts_)
        hits.append(hits_)
        sns_response.append(sns_response_)
        sns_positions.append(sns_positions_)
        evtmap.append(evtmap_)

    counter = counter+1

    if counter == 39 or index == tot_files-1:
        print("Writing file out")

        reco          = pd.concat(reco)
        dst           = pd.concat(dst)
        runevts       = pd.concat(runevts)
        runinfo       = pd.concat(runinfo)

        if (run_number <0):

            config        = pd.concat(config)
            parts         = pd.concat(parts)
            hits          = pd.concat(hits)
            sns_response  = pd.concat(sns_response)
            sns_positions = pd.concat(sns_positions)
            evtmap        = pd.concat(evtmap)

            # Open the HDF5 file in write mode
            with pd.HDFStore(f"230725/combined/sophronia_{out_basename}_{index}.h5", mode='w', complevel=5, complib='zlib') as store:
                # Write each DataFrame to the file with a unique key
                store.put('RECO/Events',      reco,    format='table')
                store.put('DST/Events',       dst,     format='table')
                store.put('Run/events',       runevts, format='table')
                store.put('Run/runInfo',      runinfo, format='table')

                store.put('MC/configuration', config,        format='table')
                store.put('MC/particles',     parts,         format='table')
                store.put('MC/hits',          hits,          format='table')
                store.put('MC/sns_response',  sns_response,  format='table')
                store.put('MC/sns_positions', sns_positions, format='table')
                store.put('Run/eventMap',     evtmap,        format='table')

        else:

            # Open the HDF5 file in write mode
            with pd.HDFStore(f"230725/combined/sophronia_{out_basename}_{index}.h5", mode='w', complevel=5, complib='zlib') as store:
                # Write each DataFrame to the file with a unique key
                store.put('RECO/Events',      reco,    format='table')
                store.put('DST/Events',       dst,     format='table')
                store.put('Run/events',       runevts, format='table')
                store.put('Run/runInfo',      runinfo, format='table')

        print(reco)
        print(dst)
        print("Number of events in new file:", len(reco.event.unique()))

        # Reset 
        counter = 0
        reco          = []
        dst           = []
        runevts       = []
        runinfo       = []
        config        = []
        parts         = []
        hits          = []
        sns_response  = []
        sns_positions = []