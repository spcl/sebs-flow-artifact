import pandas as pd 
import os

dfs = []

reps = range(1,7)
for rep in reps:
  df = pd.read_csv(os.path.join("rep" + str(rep), "burst_2048.csv"))
  dfs.append(df)
  print(len(df))
  
df = pd.concat(dfs)
df.to_csv("burst_2048.csv")
