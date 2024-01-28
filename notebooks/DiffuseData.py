# Script to read in the nexus files and diffuse them 1 electron at a time
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

# Diffusion parameters
DL = 0.415 # mm / sqrt(cm)
DT = 0.316 # mm / sqrt(cm)

# Create the bins ---- 
xmin=-3000
xmax=3000
xbw=10

ymin=-3000
ymax=3000
ybw=10

zmin=0
zmax=6000
zbw=10 # 100 ns timing with vd = 1.18 mm/us KM: changed to see if it helps

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
def generate_random(row):
    x = row['x'] # mm
    y = row['y'] # mm
    z = row['z'] # mm
    sigma_DL = DL*np.sqrt(z/10.) # mm - Need to check this Eqn. 
    sigma_DT = DT*np.sqrt(z/10.) # mm - Need to check this Eqn. 

    xy = np.array([x, y])
    cov_xy = np.array([[sigma_DT, 0], [0, sigma_DT]])
    
    xy_smear = rng.multivariate_normal(xy, cov_xy, 1)
    z_smear = rng.normal(z, sigma_DL)

    return pd.Series([xy_smear[0, 0], xy_smear[0, 1], z_smear], index=['x_smear', 'y_smear', 'z_smear'])

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

    # Create a new DataFrame with duplicated rows, so we can smear each electron by diffusion
    electrons = pd.DataFrame(np.repeat(event[["event_id",'x', 'y', 'z']].values, event['n'], axis=0), columns=["event_id",'x', 'y', 'z'])

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