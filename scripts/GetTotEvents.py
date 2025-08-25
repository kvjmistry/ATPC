import pandas as pd
import glob
import sys

Type=sys.argv[1]
P=sys.argv[2]
D=sys.argv[3]

path = f"{Type}/{P}/{D}/*.h5"
print(path)

files = glob.glob(path)

events = 0

for f in files:
  hits = pd.read_hdf(f, "MC/hits")
  n=len(hits.event_id.unique())
  events = events+n


print("Tot events:", events)



