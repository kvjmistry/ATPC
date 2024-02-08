# Script to read in the nexus files and smear the hits along the track.
# Re-bin the resultant track
import sys
import numpy  as np
import pandas as pd
from collections import Counter
import time

# Record the start time
start_time = time.time()

# Load in the hits
print("Loading hits")
print("Filename: ", sys.argv[1]+".h5")
hits = pd.read_hdf(sys.argv[1]+".h5", 'MC/hits')
print("Finished loading hits")

# init the RNG
rng = np.random.default_rng()

# Create the bins ---- 
xmin=-3000
xmax=3000
xbw=0.25

ymin=-3000
ymax=3000
ybw=0.25

zmin=0
zmax=6000
zbw=0.25 

# bins for x, y, z
xbins = np.arange(xmin, xmax+xbw, xbw)
ybins = np.arange(ymin, ymax+ybw, ybw)
zbins = np.arange(zmin, zmax+zbw, zbw)

# center bins for x, y, z
xbin_c = xbins[:-1] + xbw / 2
ybin_c = ybins[:-1] + ybw / 2
zbin_c = zbins[:-1] + zbw / 2


df_smear = []

# Define a function to generate random numbers from Gaussian distribution
# Define a function to smear the geant4 hits uniformly between the steps
def generate_random(row):
    r0 = np.array([row['x'], row['y'], row['z']])
    r1 = np.array([row['x'] - row['dx1']/2.0, row['y'] - row['dy1']/2.0, row['z'] - row['dz1']/2.0])
    r2 = np.array([row['x'] + row['dx2']/2.0, row['y'] + row['dy2']/2.0, row['z'] + row['dz2']/2.0])
    
    # This helps us to either move to the step forward or backward from the hit
    sampled_direction = np.random.choice([1, 2], p=[0.5, 0.5])

    if (sampled_direction == 1):
        random_number = rng.uniform(0, 1)
        new_r = r0+random_number*(r1 - r0)
    else:
        random_number = rng.uniform(0, 1)
        new_r = r0+random_number*(r2 - r0)

    x_smear, y_smear, z_smear = new_r[0], new_r[1], new_r[2]
    return pd.Series([x_smear, y_smear, z_smear], index=['x_smear', 'y_smear', 'z_smear'])

# Print the number of events:
print("Number of events to process: ", len(hits.event_id.unique()))
min_event_id = min( hits.event_id.unique())

# Main event Loop
for index, e in enumerate(hits.event_id.unique()):
    print("On Event:", e - min_event_id)

    # Record the end time
    end_time = time.time()

    # Calculate and print the runtime
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.4f} seconds")

    # Select the event
    event = hits[hits.event_id == e]
    
    # Shift z-values so 0 is at the anode
    event.z = event.z+3000

    # Calc number of electrons in a hit
    event["n"] = round(event["energy"]/25e-6)

    # Loop over the particles and get the differences between steps ------
    particles = event.particle_id.unique()

    smear_df = []

    for idx, p in enumerate(particles):
        temp_part = event[event.particle_id == p]

        nrows = len(temp_part)

        # This dataframe contains the difference between hits
        diff_df = temp_part[['x', 'y', 'z']].diff()
        diff_df.iloc[0] = 0
        extra_row = pd.DataFrame({'x': [0], 'y': [0], 'z': [0]})
        diff_df = pd.concat([diff_df, extra_row])
        diff_df = diff_df.rename(columns={'x': 'dx', 'y': 'dy', 'z': 'dz'})
        
        dx1= []
        dy1 = []
        dz1 = []
        dx2= []
        dy2 = []
        dz2 = []
        index_arr = []

        # convert the difference dataframe to 
        for index in range(len(diff_df)-1):
            dx1.append(diff_df.iloc[index].dx)
            dy1.append(diff_df.iloc[index].dy)
            dz1.append(diff_df.iloc[index].dz)
            dx2.append(diff_df.iloc[index+1].dx)
            dy2.append(diff_df.iloc[index+1].dy)
            dz2.append(diff_df.iloc[index+1].dz)


        index_arr = diff_df.index.to_numpy()
        index_arr = index_arr[:-1]

        data = {
            'dx1': dx1,
            'dx2': dx2,
            'dy1': dy1,
            'dy2': dy2,
            'dz1': dz1,
            'dz2': dz2,
        }

        new_df = pd.DataFrame(data, index=index_arr)
        smear_df.append(new_df)

    # Concatenate DataFrames along rows (axis=0)
    smear_df = pd.concat(smear_df)

    # Now merge to the main df
    event = pd.merge(event, smear_df, left_index=True, right_index=True, how='inner')

    # Create a new DataFrame with duplicated rows, so we can smear each electron by diffusion
    electrons = pd.DataFrame(np.repeat(event[["event_id",'x', 'y', 'z', 'dx1', 'dx2', 'dy1', 'dy2', 'dz1','dz2']].values, event['n'], axis=0), columns=["event_id",'x', 'y', 'z', 'dx1', 'dx2', 'dy1', 'dy2', 'dz1','dz2'])

    # Reset the index of the new DataFrame if needed
    electrons = electrons.reset_index(drop=True)

    # Now apply some smearing to each of the electrons
    # Apply the function to create new columns
    new_columns = electrons.apply(generate_random, axis=1)
    electrons_smear = pd.concat([electrons, new_columns], axis=1)
    electrons_smear["energy"] = 25e-6 # MeV

    # Now lets bin the data
    electrons_smear['x_smear'] = pd.cut(x=electrons_smear['x_smear'], bins=xbins,labels=xbin_c, include_lowest=True)
    electrons_smear['y_smear'] = pd.cut(x=electrons_smear['y_smear'], bins=ybins,labels=ybin_c, include_lowest=True)
    electrons_smear['z_smear'] = pd.cut(x=electrons_smear['z_smear'], bins=zbins,labels=zbin_c, include_lowest=True)

    # Merge any duplicate rows and sum their energy
    # also rename everything back to normal x,y,z
    electrons_smear = electrons_smear.drop(columns=['x', 'y', 'z'])
    electrons_smear = electrons_smear.rename(columns={'x_smear': 'x'})
    electrons_smear = electrons_smear.rename(columns={'y_smear': 'y'})
    electrons_smear = electrons_smear.rename(columns={'z_smear': 'z'})

    # Create a list of tuples representing each row in the DataFrame
    rows_as_tuples = [tuple(row) for row in electrons_smear.values]

    # Use Counter to count the occurrences of each row
    row_counts = Counter(rows_as_tuples)

    # Map the counts back to the DataFrame
    electrons_smear['duplicates'] = [row_counts[tuple(row)] for row in electrons_smear.values]

    # Multiply 'energy' and 'duplicates' columns
    electrons_smear['energy'] = electrons_smear['energy'] * electrons_smear['duplicates']

    # Drop the 'duplicates' column
    electrons_smear.drop(columns=['duplicates'], inplace=True)

    electrons_smear = electrons_smear.drop_duplicates()

    # File writing
    electrons_smear = electrons_smear.sort_values(by=['event_id', 'z', 'x', 'y'])

    electrons_smear['event_id'] = electrons_smear['event_id'].astype(int)
    electrons_smear['z'] = electrons_smear['z'].astype('float32')
    electrons_smear['x'] = electrons_smear['x'].astype('float32')
    electrons_smear['y'] = electrons_smear['y'].astype('float32')
    # df['energy'] = df['energy']*1e6
    electrons_smear['energy'] = electrons_smear['energy'].astype('float32')

    df_smear.append(electrons_smear)


df_smear_merge = pd.concat(df_smear, ignore_index=True)

print("Saving events to file: ", sys.argv[1]+"_smear.h5")
with pd.HDFStore(sys.argv[1]+"_smear.h5", mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('hits', df_smear_merge, format='table')

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")