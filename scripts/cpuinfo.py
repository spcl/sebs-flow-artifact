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
benchmark_path = os.path.join("../perf-cost", "660.map-reduce", "gcp", "py-cpuinfo-batch-size-30-reps-1", "burst_256_processed.csv")
df = pd.read_csv(benchmark_path)

#TODO find out which entry inside cpuinfo differs and then find out if a specific cputype is correlated to low execution times? 

print("entries in df: ", len(df))
print("unique cpuinfos: ", len(df['cpuinfo'].unique()))
info_entry = df['cpuinfo'].replace("\'", "\"")
#print(info_entry[0]['python_version'])

#not_interesting_entries = ['hz_advertised_friendly', 'hz_actual_friendly', 'hz_advertised', 'hz_actual']
not_interesting_entries = []

json_infos = []
differences = {}

sort_after = 'hz_actual_friendly'

df[sort_after] = ""
df['times'] = df['end'] - df['start']
for idx, complete_entry in df.iterrows(): 
  entry = complete_entry['cpuinfo'].replace("\'", "\"")
  entry = json.loads(entry)
  #complete_entry['flags'] = entry['flags']

  #FIXME specific for hertz.
  hertz = entry[sort_after]
  hertz = hertz.replace(" GHz", "")
  df.at[idx, sort_after] = float(hertz)
  
  #df.at[idx, sort_after] = str(entry[sort_after])
  if len(json_infos) == 0:
    json_infos.append(entry)
  if entry not in json_infos:
    for existing_info in json_infos:
      for single_entry in entry:
        #print(entry[single_entry])
        if single_entry in not_interesting_entries:
          #skip
          continue
        if entry[single_entry] != existing_info[single_entry]:
          #print("differing: ", single_entry, entry[single_entry], "existing: ", existing_info[single_entry])
          if entry not in json_infos:
            json_infos.append(entry)
          if single_entry not in differences.keys():
            differences[single_entry] = []
          if entry[single_entry] not in differences[single_entry]:
            differences[single_entry].append(entry[single_entry])
            
print(differences)
print(len(differences['flags']))
print(len(json_infos))

dfs= []
# do the flags correlate with execution time? 

#for idx, flags in enumerate(differences[sort_after]):
hertz = []
hertz.append(2.5)
#for idx, flags in hertz:
print(df[sort_after].replace(" GHz", ""))
entries1 = df.loc[df[sort_after] <= 2.5]
entries2 = df.loc[df[sort_after] > 2.5]

#entries = df.loc[df[sort_after] == str(flags)]
funcs = entries1.groupby("func")
for key, item in funcs:
  my_df = funcs.get_group(key)["times"]
  print("func: ", key, my_df.mean(), "length: ", len(my_df))
  #now write to new df
  new_df = pd.DataFrame(my_df, columns=["times"])
  exp_id = ""
  exp_id += sort_after + "\nfunc: " + key + "hz_actual_friendly <= 2.5GHz" + "\nmean " + str(round(new_df['times'].mean(),2))
  new_df['exp_id'] = exp_id

  dfs.append(new_df)
  
funcs = entries2.groupby("func")
for key, item in funcs:
  my_df = funcs.get_group(key)["times"]
  print("func: ", key, my_df.mean(), "length: ", len(my_df))
  #now write to new df
  new_df = pd.DataFrame(my_df, columns=["times"])
  exp_id = ""
  exp_id += sort_after + "\nfunc: " + key + "hz_actual_friendly > 2.5GHz" + "\nmean " + str(round(new_df['times'].mean(),2))
  new_df['exp_id'] = exp_id

  dfs.append(new_df)
      
df = pd.concat(dfs)
df = df.sort_values('exp_id')

#TODO
#find out what min and max of hz ist
print("min hz_actual_friendly", min(differences['hz_actual_friendly']), "max ", max(differences['hz_actual_friendly']))

title = "hz_actual_friendly from ", min(differences['hz_actual_friendly']), "to ", max(differences['hz_actual_friendly'])
plt.title(title)

#my_pal = {exp_id: randomcolor(exp_id) for exp_id in df["exp_id"].unique()}
#my_pal = {exp_id: randomcolor(exp_id.split('\n')[0]) for exp_id in df["exp_id"].unique()}
sb.violinplot(x="exp_id", y="times", data=df, cut=0) 
#sb.violinplot(x="exp_id", y="times", data=df, cut=0) #, palette=my_pal)



ax.set_ylabel("Duration [s]")
ax.set_xlabel(None)
plt.show()
