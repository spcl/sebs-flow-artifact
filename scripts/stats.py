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
  benchmark_path = os.path.join("../perf-cost", args.benchmark, platform, config, "burst_" + args.memory + ".csv")
  print("benchmark path: ", benchmark_path)
  complete_df = pd.read_csv(benchmark_path)
  complete_df["times"] = complete_df["end"] - complete_df["start"]

  funcs = complete_df['func'].unique()
  for func in funcs:
    df = complete_df.loc[complete_df['func'] == func]
    invos = df.groupby("func")


    variations = []
    
    for key, item in invos:
        my_df = invos.get_group(key)["times"]
        variation = scipy.stats.variation(my_df)
        if variation != 0:
          variations.append(scipy.stats.variation(my_df))
    if len(variations) >0: 
      print("func: ", func, "mean variation", round(sum(variations)/float(len(variations)),2), "mean duration", round(my_df.mean(),2), "min variation: ", round(min(variations),2), "max: ", round(max(variations),2))
