# Script to store all the common reconstruction functions used in the analysis. 
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------
def ApplyCuts(df_meta, df_primary, pressure, diffusion, mode, Eres):

    cuts = []

    # Apply containment
    df_meta = df_meta[df_meta.contained == True]
    df_primary = df_primary[df_primary.contained == True]
    df_meta, df_primary = ApplyEventEnergyCut(df_meta, df_primary, Eres)

    # This is common for all NEXT-tonne analysis
    if (mode == "next1t"):
        df_counts = GetNTracks(df_meta)
        df_counts = df_counts[df_counts.N_tracks == 1]
        df_meta = df_meta[df_meta["event_id"].isin(df_counts.event_id.unique())]
        df_primary = df_primary[df_primary["event_id"].isin(df_counts.event_id.unique())]

    # 1 bar
    if (pressure == 1):
       
        # 1bar no diff
        if (diffusion == "nodiff"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.33) & (df_primary.energy > 2.37)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.44) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.0)
            else:
                print("Unknown cut mode specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 1bar 5%
        elif (diffusion == "5percent"):
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.33) &  (df_primary.energy > 2.38) 
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.44) & (df_primary.blob2 > 0.45) & (df_primary.blob1R > 0.34) &  (df_primary.energy > 2.35) & (df_primary.Tortuosity2 >1.3)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.0)
            else:
                print("Unknown cut mode specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 1bar 0.25%
        elif (diffusion == "0.25percent"):
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.42) & (df_primary.blob2 > 0.45) & (df_primary.energy > 2.39)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.45) & (df_primary.blob2 > 0.54) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.35)
            else:
                print("Unknown cut mode specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 1bar 0.1%
        elif (diffusion == "0.1percent"):
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.42) & (df_primary.blob2 > 0.45) & (df_primary.energy > 2.4) & (df_primary.length > 1000)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.44) & (df_primary.blob2 > 0.6)  & (df_primary.energy > 2.4) & (df_primary.length > 1000)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.35)
            else:
                print("Unknown cut mode specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 1bar 10 He%
        elif (diffusion == "0.05percent"):
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.41) & (df_primary.blob2 > 0.4) & (df_primary.energy > 2.4) & (df_primary.length > 1000)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.48) & (df_primary.blob2 > 0.5) & (df_primary.energy > 2.4) & (df_primary.length > 600)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.37)
            else:
                print("Unknown cut mode specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 1bar 0.0%
        elif (diffusion == "0.0percent"):
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2 > 0.18) & (df_primary.blob2R > 0.27) &  (df_primary.energy > 2.33) & (df_primary.length > 600)
            elif (mode == "nat"):
                cuts = (df_primary.blob2 > 0.35) & (df_primary.blob2R > 0.38) &  (df_primary.blob1 > 0.55) &  (df_primary.energy > 2.4) & (df_primary.length > 600)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.16)
            else:
                print("Unknown cut mode specified")

        else:
            print("Unknown diffusion specified")
    # -------------------------------------------------------------------------------------------------------------------------------------
    # 5bar
    elif (pressure == 5):
    
        # 5 bar no diff
        if (diffusion == "nodiff"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.42) & (df_primary.blob2 > 0.495) & (df_primary.energy > 2.4)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.45) & (df_primary.blob2 > 0.51) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.46)
            else:
                print("Unknown efficiency target specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 5bar 5%
        elif (diffusion == "5percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.52) & (df_primary.energy > 2.4)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.4) &  (df_primary.blob2 > 0.56) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.445)
            else:
                print("Unknown cut mode specified")

        # 5bar 10% He
        elif (diffusion == "0.05percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2 > 0.67) & (df_primary.energy > 2.35) & (df_primary.blob1R > 0.6) 
            elif (mode == "nat"):
                cuts = (df_primary.blob2 > 0.71) & (df_primary.energy > 2.4) & (df_primary.length > 200)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.42)
            else:
                print("Unknown cut mode specified")

        else:
            print("Unknown diffusion specified")

    # -------------------------------------------------------------------------------------------------------------------------------------
    # 10bar
    elif (pressure == 10):

        # 10bar no diff
        if (diffusion == "nodiff"):
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.4) & (df_primary.blob2 > 0.58) & (df_primary.energy > 2.4)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.5) & (df_primary.blob2 > 0.57) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.525)
            else:
                print("Unknown cut mode specified")

        # -------------------------------------------------------------------------------------------------------------------------------------
        # 10bar 5%
        elif (diffusion == "5percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.555) & (df_primary.energy > 2.4) 
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.53) & (df_primary.blob2 > 0.53) &  (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.515)
            else:
                print("Unknown cut mode specified")

        # 10bar 10% He
        elif (diffusion == "0.05percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.44) & (df_primary.blob2 > 0.27) & (df_primary.energy > 2.37) 
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.45) & (df_primary.blob2 > 0.47) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.415)
            else:
                print("Unknown cut mode specified")

        else:
            print("Unknown diffusion specified")

    # -------------------------------------------------------------------------------------------------------------------------------------
    # 15bar
    elif (pressure == 15):
        
        # 15bar no diff
        if (diffusion == "nodiff"):
        
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.52) & (df_primary.blob2 > 0.61) & (df_primary.energy > 2.4)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.56) & (df_primary.blob2 > 0.58) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.562)
            else:
                print("Unknown cut mode specified")

        # -------------------------------------------------------------------------------------------------------------------------------------
        # 15bar 5%
        elif (diffusion == "5percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.6) & (df_primary.energy > 2.37)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.54) & (df_primary.blob2 > 0.55) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.56)
            else:
                print("Unknown cut mode specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 15bar 10% he
        elif (diffusion == "0.05percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts =  (df_primary.blob2 > 0.73) & (df_primary.energy > 2.38)
            elif (mode == "nat"):
                cuts =  (df_primary.blob2 > 0.75) & (df_primary.energy > 2.42)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.41)
            else:
                print("Unknown cut mode specified")

        else:
            print("Unknown diffusion specified")

    # -------------------------------------------------------------------------------------------------------------------------------------
    # 25bar
    elif (pressure == 25):

        # 25bar no diff
        if (diffusion == "nodiff"):
        
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.56) & (df_primary.blob2 > 0.695) & (df_primary.energy > 2.4)
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.57) & (df_primary.blob2 > 0.7) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.62)
            else:
                print("Unknown cut mode specified")

        # -------------------------------------------------------------------------------------------------------------------------------------
        # 25bar 5%
        elif (diffusion == "5percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.53) & (df_primary.blob2 > 0.43) & (df_primary.blob1 > 0.3) & (df_primary.energy > 2.4) 
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.45) & (df_primary.blob2 > 0.55) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.58)
            else:
                print("Unknown cut mode specified")
        # -------------------------------------------------------------------------------------------------------------------------------------
        # 25bar 10% He
        elif (diffusion == "0.05percent"):
            
            # All cuts
            if (mode == "enr"):
                cuts = (df_primary.blob2R > 0.43) & (df_primary.blob2 > 0.43) & (df_primary.blob1 > 0.3) & (df_primary.energy > 2.4) 
            elif (mode == "nat"):
                cuts = (df_primary.blob2R > 0.45) & (df_primary.blob2 > 0.5) & (df_primary.energy > 2.4)
            elif (mode == "next1t"):
                cuts = (df_primary.blob2R > 0.44)
            else:
                print("Unknown cut mode specified")

        else:
            print("Unknown diffusion specified")

    else:
        print("Unknown pressure specified")

    return df_meta, df_primary, cuts
# ------------------------------------------------------------------------
# Function to get the number of tracks (excluding deltas)
def GetNTracks(df_meta):

    # Get all tracks without the name delta
    filtered_df = df_meta[~df_meta["label"].str.contains("Delta", na=False)]

    # Step 2: Count unique trkID per event_id
    trk_counts = filtered_df.groupby("event_id")["trkID"].nunique().reset_index()
    trk_counts.rename(columns={"trkID": "N_tracks"}, inplace=True)

    return trk_counts

# ------------------------------------------------------------------------
def GetNParticles(df_meta, label):
    # Set threshold for cumulative_distance
    threshold = 0

    # Filter events with brem in the name
    filtered_df = df_meta[df_meta["label"].str.contains(label, na=False)]

    # Step 2: Count unique trkID per event_id
    trk_counts = filtered_df.groupby("event_id")["trkID"].nunique().reset_index()
    trk_counts.rename(columns={"trkID": f"N_{label}"}, inplace=True)

    return trk_counts
# ------------------------------------------------------------------------

def GetDeltas(df_meta):
    filtered_df = df_meta[df_meta["label"].str.contains("Delta", na=False)]

    return filtered_df

# ------------------------------------------------------------------------
def GetLargestDelta(df_meta):
    filtered_df = df_meta[df_meta["label"].str.contains("Delta", na=False)]

    # Get the max energy value per event
    max_energy_per_event = filtered_df.groupby("event_id")["energy"].transform("max")

    # Keep only rows where energy matches the max per event
    return filtered_df[filtered_df["energy"] == max_energy_per_event]

# ------------------------------------------------------------------------
def GetBrems(df_meta):
    filtered_df = df_meta[df_meta["label"].str.contains("Brem", na=False)]

    return filtered_df

# ------------------------------------------------------------------------
# Gets the sum of the primary track and the delta energies
def GetTrackDeltaEnergy(df_meta):
    # Filter events with brem in the name
    filtered_df = df_meta[ ~df_meta["label"].str.contains("Brem", na=False)]
    filtered_df =filtered_df[filtered_df.energy>0.01]
    trk_energies = filtered_df.groupby(["event_id"])["energy"].sum()
    return trk_energies.values
# ------------------------------------------------------------------------
def FOM(eff, bkg_eff):
    return eff/np.sqrt(bkg_eff)

# ------------------------------------------------------------------------
def Calc_FOM_err(fom, eff, eff_err, bkg, bkg_err):
    return fom*np.sqrt( (eff_err/eff)**2 + 0.25*(bkg_err/bkg)**2)

# ------------------------------------------------------------------------
def CalcEfficiency(n, N, label, pressure, mass):
    
    efficiency = n/N
    error=np.sqrt( (efficiency/N) * (1-efficiency)  )
    
    if (label == "nubb"):
        efficiency = ApplyContainmentCorr(efficiency, pressure, mass)
        print(f"Efficiency {label}:",  round(100*efficiency, 3), " +/-", round(100*error,3),  "%")
    else:
        print(f"Bkg Rej: {label}:", round(100*efficiency, 3), "+/-", round(100*error,3),  "%", "     (bkg rej ==", round(100-100*efficiency,3), "%)")

    return efficiency, error
# ------------------------------------------------------------------------
def ApplyAssymEResCut(label, n):

    if (label == "nubb"):
        return 0.82*n
    elif (label == "Bkg"):
        return 0.4*n
    elif (label == "Bi"):
        return 0.38*n
    elif (label =="Tl"):
        return 0.5*n
    elif (label == "Single"):
        return 0.53*n
    else:
        return 1.0*n

# ------------------------------------------------------------------------
# Function to compute count-based ratio per event_id
def compute_ratio(group):
    # Group 1: Count of Primary + Delta*
    count_group1 = group[group["label"].str.startswith(("Primary", "Delta"))].shape[0]

    # Group 2: Count of Brem* where 0.025 < energy < 0.035 (X-rays)
    count_xrays = group[(group["label"].str.startswith("Brem")) & (group["energy"].between(0.025, 0.035))].shape[0]

    # Group 3: Count of Brem* where energy ≤ 0.025 or ≥ 0.035
    count_group3 = group[(group["label"].str.startswith("Brem")) & (~group["energy"].between(0.025, 0.035))].shape[0]

    # Compute ratio
    ratio = count_xrays / (count_group1 + count_group3)
    
    return ratio

# ------------------------------------------------------------------------
# applies cuts to the delta, brem and primary+delta energies
def ApplyGeneralCuts(df_meta, df_primary, cut_brem, cut_delta, cut_trk_e):

    brems  = GetBrems(df_meta)

    brems = brems[brems.energy > cut_brem] # these are events to cut

    df_meta = df_meta[~df_meta.event_id.isin(brems.event_id.unique())]
    df_primary = df_primary[~df_primary.event_id.isin(brems.event_id.unique())]

    deltas  = GetDeltas(df_meta)

    deltas = deltas[deltas.energy > cut_delta] # these are events to cut

    df_meta = df_meta[~df_meta.event_id.isin(deltas.event_id.unique())]
    df_primary = df_primary[~df_primary.event_id.isin(deltas.event_id.unique())]

    filtered_df = df_meta[ ~df_meta["label"].str.contains("Brem", na=False)]
    filtered_df =filtered_df[filtered_df.energy>0.01]
    trk_energies = filtered_df.groupby(["event_id"])["energy"].sum()
    trk_energies = trk_energies[trk_energies > cut_trk_e]
    unique_events_list = trk_energies.index.unique().tolist()

    df_meta = df_meta[~df_meta.event_id.isin(unique_events_list)]
    df_primary = df_primary[~df_primary.event_id.isin(unique_events_list)]

    return df_meta, df_primary

# ------------------------------------------------------------------------
def ApplyDeltaLenCut(df_meta, df_primary, cut_delta):

    deltas  = GetDeltas(df_meta)

    deltas = deltas[deltas.length > cut_delta] # these are events to cut

    df_meta    = df_meta[~df_meta.event_id.isin(deltas.event_id.unique())]
    df_primary = df_primary[~df_primary.event_id.isin(deltas.event_id.unique())]

    return df_meta, df_primary
# ------------------------------------------------------------------------

# Cut out events with Brem in the name
def ApplyNTracksLenCut(df_meta, df_primary, keep_xrays):

    brems  = GetBrems(df_meta)

    if (keep_xrays):
        brems = brems[ (brems.energy > 30e-3) ] # these are events to cut

    df_meta    = df_meta[~df_meta.event_id.isin(brems.event_id.unique())]
    df_primary = df_primary[~df_primary.event_id.isin(brems.event_id.unique())]

    return df_meta, df_primary
# ------------------------------------------------------------------------

# Cut out events with Brem in the name
def ApplyEventEnergyCut(df_meta, df_primary, Eres):

    event_energy = df_meta.groupby("event_id").energy.sum()

    if (Eres == 0.5):
        good_events = event_energy[(event_energy >= 2.454) & (event_energy <= 2.471)].index # 0.5%
    else:
        good_events = event_energy[(event_energy >= 2.433) & (event_energy <= 2.48)].index # 1.0%
    
    df_meta = df_meta[df_meta["event_id"].isin(good_events)]
    df_primary = df_primary[df_primary["event_id"].isin(good_events)]

    return df_meta, df_primary

# ------------------------------------------------------------------------
def ApplyNTrackCut(df_meta, df_primary, n_track):

    # Set threshold for cumulative_distance
    threshold = 0

    # Step 1: Filter rows based on cumulative_distance
    filtered_df = df_meta[df_meta["length"] >= threshold]

    # Step 2: Count unique trkID per event_id
    trk_counts = filtered_df.groupby("event_id")["trkID"].nunique().reset_index()
    trk_counts.rename(columns={"trkID": "N_tracks"}, inplace=True)
    trk_counts = trk_counts[trk_counts.N_tracks <= n_track]

    df_meta    = df_meta[df_meta.event_id.isin(trk_counts.event_id.unique())]
    df_primary = df_primary[df_primary.event_id.isin(trk_counts.event_id.unique())]

    return df_meta, df_primary

# ------------------------------------------------------------------------
def ApplyContainmentCorr(eff, p, mass):

    if (p == 1):
        if mass == 1:
            factor = 0.532
        else:
            factor = 0.690
        print("Correcting Efficiency by factor ", factor)
        return eff*factor
    elif (p == 5):
        if mass == 1:
            factor = 0.740
        else:
            factor = 0.828
        print("Correcting Efficiency by factor ", factor)
        return eff*factor
    elif (p == 10):
        if mass == 1:
            factor = 0.806
        else:
            factor = 0.872
        print("Correcting Efficiency by factor ", factor)
        return eff*factor
    elif (p == 15):
        if mass == 1:
            factor = 0.835
        else:
            factor = 0.898
        print("Correcting Efficiency by factor ", factor)
        return eff*factor
    elif (p == 25):
        if mass == 1:
            factor = 0.876
        else:
            factor = 0.92
        print("Correcting Efficiency by factor ", factor)
        return eff*factor
    else:
        return eff

# ------------------------------------------------------------------------

def PlotDistributions(df_meta, col, label, pressure, diffusion, mode, Eres, scale_factor, axs, applycuts):

    print("Running cuts with")
    print("Pressure:",  pressure,"bar")
    print("Diffusion:", diffusion )

    uselog=True
    # uselog=False

    df_primary = df_meta[ (df_meta.label == "Primary") & (df_meta.primary == 1)]

    # Apply the cuts
    if applycuts:
        df_meta, df_primary, cuts = ApplyCuts(df_meta, df_primary, pressure, diffusion, mode, Eres)
        df_primary = df_primary[ cuts ]
        df_meta = df_meta[(df_meta.event_id.isin(df_primary.event_id.unique()))]

    event_energy = df_meta.groupby("event_id").energy.sum()

    df_counts      = GetNTracks(df_meta)
    df_counts_evts = df_counts[df_counts.N_tracks >= 1].event_id.unique()

    # Number of brems and deltas
    N_brem  = GetNParticles(df_meta, "Brem")
    N_delta = GetNParticles(df_meta, "Delta")

    # Dataframes containing brems and deltas
    deltas = GetDeltas(df_meta)
    brems  = GetBrems(df_meta)

    # This is the sum of the primary and delta energies attached to it
    trk_e = GetTrackDeltaEnergy(df_meta)

    # Calculate the ratio of x-rays to tracks
    # x_ray_ratio = df_meta.groupby("event_id").apply(compute_ratio).reset_index(name="ratio")


    weights        = np.ones_like(df_primary.energy)  * scale_factor
    weights_counts = np.ones_like(df_counts.N_tracks) * scale_factor
    weights_deltas = np.ones_like(deltas.energy)      * scale_factor
    weights_brem   = np.ones_like(brems.energy)       * scale_factor
    weights_Ndelta = np.ones_like(N_delta.N_Delta)    * scale_factor
    weights_Nbrem  = np.ones_like(N_brem.N_Brem)      * scale_factor
    weights_trke   = np.ones_like(trk_e)              * scale_factor
    weights_energy = np.ones_like(event_energy)       * scale_factor

    bin_edges = np.arange(-0.5, 7.5, 1)
    bin_centers = np.arange(0, 7, 1)

    # Multiplicities
    axs[0,0].hist(df_counts.N_tracks, bins = bin_edges, histtype="step", color = col, label = label, weights = weights_counts);
    # axs[0,0].hist(N_brem.N_Brem, bins = bin_edges, histtype="step", color = col, label = label);
    axs[0,0].set_xlabel("N Tracks per event")
    axs[0,0].set_ylabel("Entries")
    axs[0,0].set_xticks(bin_centers) ;
    axs[0,0].legend()
    if (uselog): axs[0,0].semilogy()

    axs[0,1].hist(df_primary.length, bins = np.linspace(0, 5000/pressure, 100), histtype="step", color = col, label = label, weights = weights);
    axs[0,1].set_xlabel("Primary Track Length / P [mm/bar]")
    axs[0,1].set_ylabel("Entries")
    axs[0,1].legend()
    if (uselog): axs[0,1].semilogy()

    axs[0,2].hist(df_primary.energy, bins = np.linspace(0,3,100), histtype="step", color = col, label = label, weights = weights);
    axs[0,2].set_xlabel("Primary Track Energy [MeV]")
    axs[0,2].set_ylabel("Entries")
    axs[0,2].legend()
    if (uselog): axs[0,2].semilogy()

    axs[1,1].hist(df_primary.blob1, bins = np.linspace(0, 2, 100), histtype="step", color = col, label = label, weights = weights);
    axs[1,1].set_xlabel("Blob 1 energy [MeV]")
    axs[1,1].set_ylabel("Entries")
    axs[1,1].legend()
    if (uselog): axs[1,1].semilogy()

    axs[2,1].hist(df_primary.blob2, bins = np.linspace(0, 2, 100), histtype="step", color = col, label = label, weights = weights);
    axs[2,1].set_xlabel("Blob 2 energy [MeV]")
    axs[2,1].set_ylabel("Entries")
    axs[2,1].legend()
    if (uselog): axs[2,1].semilogy()

    axs[1,0].hist(df_primary.blob1R, bins = np.linspace(0, 2, 100), histtype="step", color = col, label = label, weights = weights);
    axs[1,0].set_xlabel("Blob 1 energy radius [MeV]")
    axs[1,0].set_ylabel("Entries")
    axs[1,0].legend()
    if (uselog): axs[1,0].semilogy()

    axs[2,0].hist(df_primary.blob2R, bins = np.linspace(0, 1, 100), histtype="step", color = col, label = label, weights = weights);
    axs[2,0].set_xlabel("Blob 2 energy radius [MeV]")
    axs[2,0].set_ylabel("Entries")
    axs[2,0].legend()
    if (uselog): axs[2,0].semilogy()

    axs[1,2].hist(df_primary.Tortuosity1, bins = np.linspace(0, 10, 100), histtype="step", color = col, label = label, weights = weights);
    axs[1,2].set_xlabel("Tortuosity Blob 1")
    axs[1,2].set_ylabel("Entries")
    axs[1,2].legend()
    if (uselog): axs[1,2].semilogy()

    axs[2,2].hist(df_primary.Tortuosity2, bins = np.linspace(0, 6, 100), histtype="step", color = col, label = label, weights = weights);
    axs[2,2].set_xlabel("Tortuosity Blob 2")
    axs[2,2].set_ylabel("Entries")
    axs[2,2].legend()
    if (uselog): axs[2,2].semilogy()

    axs[3,0].hist(df_primary.Squiglicity2, bins = np.linspace(0, 5, 100), histtype="step", color = col, label = label, weights = weights);
    axs[3,0].set_xlabel("Squiglicity Blob 2 [MeV]")
    axs[3,0].set_ylabel("Entries")
    axs[3,0].legend()
    if (uselog): axs[3,0].semilogy()

    axs[3,1].hist(df_primary.Tortuosity1/df_primary.Tortuosity2, bins = np.linspace(0, 3, 100), histtype="step", color = col, label = label, weights = weights);
    axs[3,1].set_xlabel("Tortuosity Ends Ratio")
    axs[3,1].set_ylabel("Entries")
    axs[3,1].legend()
    if (uselog): axs[3,1].semilogy()

    axs[3,2].hist(df_primary.Squiglicity1/df_primary.Squiglicity2, bins = np.linspace(0, 5, 100), histtype="step", color = col, label = label, weights = weights);
    axs[3,2].set_xlabel("Squiglicity Ends Ratio")
    axs[3,2].set_ylabel("Entries")
    axs[3,2].legend()
    if (uselog): axs[3,2].semilogy()

    axs[4,0].hist(deltas.energy, bins = np.linspace(0,1,100), histtype="step", color = col, label = label, weights = weights_deltas);
    axs[4,0].set_xlabel("Delta Energy [MeV]")
    axs[4,0].set_ylabel("Entries")
    axs[4,0].legend()
    if (uselog): axs[4,0].semilogy()

    axs[4,1].hist(brems.energy, bins = np.linspace(0,1,100), histtype="step", color = col, label = label, weights = weights_brem);
    axs[4,1].set_xlabel("Brem Energy [MeV]")
    axs[4,1].set_ylabel("Entries")
    axs[4,1].legend()
    if (uselog): axs[4,1].semilogy()

    axs[4,2].hist(deltas.length, bins = np.linspace(0,250,100), histtype="step", color = col, label = label, weights = weights_deltas);
    axs[4,2].set_xlabel("Delta Length [mm]")
    axs[4,2].set_ylabel("Entries")
    axs[4,2].legend()
    if (uselog): axs[4,2].semilogy()

    axs[5,0].hist(brems.length, bins = np.linspace(0,250,100), histtype="step", color = col, label = label, weights = weights_brem);
    axs[5,0].set_xlabel("Brem Length [mm]")
    axs[5,0].set_ylabel("Entries")
    axs[5,0].semilogy()
    axs[5,0].legend()
    if (uselog): axs[5,0].semilogy()

    axs[5,1].hist(N_brem.N_Brem, bins = bin_edges, histtype="step", color = col, label = label, weights = weights_Nbrem);
    axs[5,1].set_xlabel("N Brem per event")
    axs[5,1].set_ylabel("Entries")
    axs[5,1].set_xticks(bin_centers) ;
    axs[5,1].legend()
    if (uselog): axs[5,1].semilogy()

    axs[5,2].hist(N_delta.N_Delta, bins = bin_edges, histtype="step", color = col, label = label, weights = weights_Ndelta);
    axs[5,2].set_xlabel("N Delta per event")
    axs[5,2].set_ylabel("Entries")
    axs[5,2].set_xticks(bin_centers) ;
    axs[5,2].legend()
    if (uselog): axs[5,2].semilogy()

    axs[6,0].hist(trk_e, bins = np.linspace(0,3,100), histtype="step", color = col, label = label, weights = weights_trke);
    axs[6,0].set_xlabel("Primary +Delta Track Energy [MeV]")
    axs[6,0].set_ylabel("Entries")
    axs[6,0].legend()
    if (uselog): axs[6,0].semilogy()

    axs[6,1].hist(event_energy, bins = np.linspace(2.3,2.6,100), histtype="step", color = col, label = label, weights = weights_energy);
    axs[6,1].set_xlabel("Event Energy [MeV]")
    axs[6,1].set_ylabel("Entries")
    axs[6,1].legend()
    if (uselog): axs[6,1].semilogy()

    # axs[6,2].hist(deltas.Tortuosity1, bins = np.linspace(0, 1, 100), histtype="step", color = col, label = label, weights = weights_deltas);
    # axs[6,2].set_xlabel("Delta Blob2 Energy")
    # axs[6,2].set_ylabel("Entries")
    # axs[6,2].legend()
    # if (uselog): axs[6,2].semilogy()

    axs[6,2].hist(df_primary.blob2RTD, bins = np.linspace(0, 2, 100), histtype="step", color = col, label = label, weights = weights);
    axs[6,2].set_xlabel("Blob 2 energy RTD [MeV]")
    axs[6,2].set_ylabel("Entries")
    axs[6,2].legend()
    if (uselog): axs[6,2].semilogy()


    bin_edges = np.arange(-0.5, 3.5, 0.5)
    bin_centers = np.arange(0, 3, 0.5)

    # Multiplicities
    # axs[6,1].hist(x_ray_ratio.ratio, bins = bin_edges, histtype="step", color = col, label = label);
    # axs[6,1].set_xlabel("N x-rays per track")
    # axs[6,1].set_ylabel("Entries")
    # axs[6,1].set_xticks(bin_centers) ;
    # axs[6,1].legend()
    # if (uselog): axs[6,1].semilogy()



    plt.tight_layout()

    return df_primary.event_id.unique()

# ------------------------------------------------------------------------
def ApplyCutsnoPlot(df_meta, pressure, diffusion, mode, E_res):
    
    df_primary = df_meta[ (df_meta.label == "Primary") & (df_meta.primary == 1)]

    # Apply the cuts
    df_meta, df_primary, cuts = ApplyCuts(df_meta, df_primary, pressure, diffusion, mode, E_res)
    df_primary = df_primary[ cuts ]

    return len(df_primary.event_id.unique())
