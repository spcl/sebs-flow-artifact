import pandas as pd
import scipy
import os
import argparse
import json

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def randomcolor(data):
    x = hash(repr(data))
    r = 25 + x % 100
    g = 25 + int(x / 0xfa) % 100
    b = 50 + int(x / 0xff) % 100
    return "#" + hex(r)[2:].rjust(2,'0') + hex(g)[2:].rjust(2,'0') + hex(b)[2:].rjust(2,'0')
    #print(len(colors))
    #return colors[x % (len(colors) - 1)]

sb.set_theme()
sb.set_context("paper")
ax = plt.gca()

#read in data
benchmark_path = os.path.join("../perf-cost", "660.map-reduce", "gcp", "batch-size-30-reps-6", "burst_256_processed.csv")
df = pd.read_csv(benchmark_path)

#TODO find out which entry inside cpuinfo differs and then find out if a specific cputype is correlated to low execution times? 

sort_after = 'is_cold'


df['times'] = df['end'] - df['start']

print(df['is_cold'])
entries1 = df.loc[df['is_cold'] == True]
entries2 = df.loc[df['is_cold'] == False]


dfs = []

funcs = df.groupby("func")
for key, item in funcs:
  my_df = funcs.get_group(key)["times"]
  print("func: ", key, my_df.mean(), "length: ", len(my_df))
  #now write to new df
  new_df = pd.DataFrame(my_df, columns=["times"])
  exp_id = ""
  exp_id += "func: " + key + "\n" + sort_after + "Both" + "\nmean " + str(round(new_df['times'].mean(),2))
  new_df['exp_id'] = exp_id

  dfs.append(new_df)

#entries = df.loc[df[sort_after] == str(flags)]
funcs = entries1.groupby("func")
for key, item in funcs:
  my_df = funcs.get_group(key)["times"]
  print("func: ", key, my_df.mean(), "length: ", len(my_df))
  #now write to new df
  new_df = pd.DataFrame(my_df, columns=["times"])
  exp_id = ""
  exp_id += "func: " + key + "\n" + sort_after + "True" + "\nmean " + str(round(new_df['times'].mean(),2))
  new_df['exp_id'] = exp_id

  dfs.append(new_df)
  
funcs = entries2.groupby("func")
for key, item in funcs:
  my_df = funcs.get_group(key)["times"]
  print("func: ", key, my_df.mean(), "length: ", len(my_df))
  #now write to new df
  new_df = pd.DataFrame(my_df, columns=["times"])
  exp_id = ""
  exp_id += "func: " + key + "\n" + sort_after + "False" + "\nmean " + str(round(new_df['times'].mean(),2))
  new_df['exp_id'] = exp_id

  dfs.append(new_df)
df = pd.concat(dfs)
df = df.sort_values('exp_id')
#my_pal = {exp_id: randomcolor(exp_id) for exp_id in df["exp_id"].unique()}
#my_pal = {exp_id: randomcolor(exp_id.split('\n')[0]) for exp_id in df["exp_id"].unique()}
sb.violinplot(x="exp_id", y="times", data=df, cut=0) 
#sb.violinplot(x="exp_id", y="times", data=df, cut=0) #, palette=my_pal)



ax.set_ylabel("Duration [s]")
ax.set_xlabel(None)
plt.show()
