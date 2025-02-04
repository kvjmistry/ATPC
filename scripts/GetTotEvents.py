import pandas as pd
import glob


Type="ATPC_Tl"
P="1bar"
D="nodiff"

path = f"{Type}/{P}/{D}/*.h5"

files = glob.glob(path)

events = 0

for f in files:
  hits = pd.read_hdf(f, "MC/hits")
  n=len(hits.event_id.unique())
  events = events+n


print("Tot events:", events)



