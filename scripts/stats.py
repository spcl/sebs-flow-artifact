import os
import argparse

import numpy as np
import pandas as pd
import scipy


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", type=str)
parser.add_argument("-p", "--platform", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-m", "--memory", default=256)
parser.add_argument("-c", "--config", nargs='+', default=[])
args = parser.parse_args()


for platform in args.platform:
  print(platform)
  #df = pd.read_csv("../perf-cost/6100.1000-genomes/gcp/burst_2048.csv")
  
  #benchmark_path = os.path.join("../perf-cost", args.benchmark, platform, "burst_" + args.memory + "_processed.csv")
  benchmark_path = os.path.join("../perf-cost", args.benchmark, platform, config, "burst_" + args.memory + ".csv")
  print("benchmark path: ", benchmark_path)
  complete_df = pd.read_csv(benchmark_path)
  complete_df["times"] = complete_df["end"] - complete_df["start"]

  funcs = complete_df['func'].unique()
  #for entry in df:
  #  if entry['func'] not in funcs:
  #    funcs.append(df['func'])
  #print(funcs)


  for func in funcs:
    df = complete_df.loc[complete_df['func'] == func]
    #invos = df.groupby("request_id")
    invos = df.groupby("func")

    #times = []
    #for entry in df: 
    #times.append(float(entry["end"] - entry["start"]))
    #print(df["times"])

    #invos = df.groupby("request_id")

    variations = []
    
    for key, item in invos:
        my_df = invos.get_group(key)["times"]
        #mini = my_df.min()
        #my_df.drop(my_df[my_df == mini].index)
        variation = scipy.stats.variation(my_df)
        if variation != 0:
          variations.append(scipy.stats.variation(my_df))
          #print("func: ", func, "variation: ", scipy.stats.variation(my_df))#, "times: ", my_df)
        #print(invos.get_group(key), "\n\n")
    if len(variations) >0: 
      print("func: ", func, "mean variation", round(sum(variations)/float(len(variations)),2), "mean duration", round(my_df.mean(),2), "min variation: ", round(min(variations),2), "max: ", round(max(variations),2))
      #print("len", len(variations))

    #print("max: ", times.max(), "min: ", times.min())

    #coeff = scipy.stats.variation(times)
    #print(coeff)

    #d_runtime = invos["end"].max() - invos["start"].min()

