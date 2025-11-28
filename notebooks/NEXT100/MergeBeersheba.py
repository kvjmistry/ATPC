import pandas as pd
import numpy as np
import glob

def read_data(infile, run_number):

    conf = pd.DataFrame()
    deco    = pd.read_hdf(infile, key = 'DECO/Events')
    dst     = pd.read_hdf(infile, key = 'DST/Events')
    filters = pd.read_hdf(infile, key = 'Filters/nohits')
    runevts = pd.read_hdf(infile, key = 'Run/events')
    runinfo = pd.read_hdf(infile, key = 'Run/runInfo')
    # conf    = pd.read_hdf(infile, key = 'config/beersheba')

    if run_number <0:
        config        = pd.read_hdf(infile, key = 'MC/configuration')
        parts         = pd.read_hdf(infile, key = 'MC/particles')
        hits          = pd.read_hdf(infile, key = 'MC/hits')
        sns_response  = pd.read_hdf(infile, key = 'MC/sns_response')
        sns_positions = pd.read_hdf(infile, key = 'MC/sns_positions')
        evtmap        = pd.read_hdf(infile, key = 'Run/eventMap')

        return deco, dst, filters, runevts, runinfo, conf, config, parts, hits, sns_response, sns_positions, evtmap
    
    return deco, dst, filters, runevts, runinfo, conf
    

out_basename = "ldc1"
files = sorted(glob.glob(f"/ospool/ap40/data/krishan.mistry/job/ATPC/Pressure/354015/{out_basename}/*.h5"))
tot_files=len(files)
run_number=99

deco          = []
dst           = []
filters       = []
runevts       = []
runinfo       = []
conf          = []
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
        deco_, dst_, filters_, runevts_, runinfo_, conf_, config_, parts_, hits_, sns_response_, sns_positions_, evtmap_ = read_data(f, run_number)
    # Data file
    else:
        deco_, dst_, filters_, runevts_, runinfo_, conf_ = read_data(f, run_number)

    deco.append(deco_)
    dst.append(dst_)
    filters.append(filters_)
    runevts.append(runevts_)
    runinfo.append(runinfo_)
    conf.append(conf_)
    
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

        deco          = pd.concat(deco)
        dst           = pd.concat(dst)
        filters       = pd.concat(filters)
        runevts       = pd.concat(runevts)
        runinfo       = pd.concat(runinfo)
        conf          = pd.concat(conf)

        if (run_number <0):

            config        = pd.concat(config)
            parts         = pd.concat(parts)
            hits          = pd.concat(hits)
            sns_response  = pd.concat(sns_response)
            sns_positions = pd.concat(sns_positions)
            evtmap        = pd.concat(evtmap)

            # Open the HDF5 file in write mode
            with pd.HDFStore(f"354015/combined/beersheba_{out_basename}_{index}.h5", mode='w', complevel=5, complib='zlib') as store:
                # Write each DataFrame to the file with a unique key
                store.put('DECO/Events',      deco,    format='table')
                store.put('DST/Events',       dst,     format='table')
                store.put('Filters/nohits',   filters, format='table')
                store.put('Run/events',       runevts, format='table')
                store.put('Run/runInfo',      runinfo, format='table')
                store.put('config/beersheba', conf,    format='table')

                store.put('MC/configuration', config,        format='table')
                store.put('MC/particles',     parts,         format='table')
                store.put('MC/hits',          hits,          format='table')
                store.put('MC/sns_response',  sns_response,  format='table')
                store.put('MC/sns_positions', sns_positions, format='table')
                store.put('Run/eventMap',     evtmap,        format='table')

        else:

            # Open the HDF5 file in write mode
            with pd.HDFStore(f"354015/combined/beersheba_{out_basename}_{index}.h5", mode='w', complevel=5, complib='zlib') as store:
                # Write each DataFrame to the file with a unique key
                store.put('DECO/Events',      deco,    format='table')
                store.put('DST/Events',       dst,     format='table')
                store.put('Filters/nohits',   filters, format='table')
                store.put('Run/events',       runevts, format='table')
                store.put('Run/runInfo',      runinfo, format='table')
                store.put('config/beersheba', conf,    format='table')

        print(deco)
        print(dst)
        print("Number of events in new file:", len(deco.event.unique()))

        # Reset 
        counter = 0
        deco          = []
        dst           = []
        filters       = []
        runevts       = []
        runinfo       = []
        conf          = []
        config        = []
        parts         = []
        hits          = []
        sns_response  = []
        sns_positions = []