import os
import argparse

import math
from typing import List, Tuple
from collections import namedtuple

import numpy as np
import pandas as pd
import seaborn as sb
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy

class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", type=str)
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-m", "--memory", nargs='+', default=[])
parser.add_argument("-c", "--config", nargs='+', default=[])
parser.add_argument("-v", "--visualization", choices=["bar", "violin", "line"], default="bar")
parser.add_argument("--sum", action="store_true", default=False)
parser.add_argument("--all", action="store_true", default=False)
args = parser.parse_args()

platform_names = {
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud",
}

colors = sb.color_palette("colorblind")
color_map = {
    "AWS": colors[0],
    "AWS Docs": colors[3],
    "Azure": colors[1],
    "Google Cloud": colors[2],
    "Google Cloud Docs": colors[4]
}

def read(path): 
    df = pd.read_csv(path)
    
    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    if np.any(df["duration"] <= 0):
        raise ValueError(f"Invalid data frame at path {path}")

    return df


def ci_le_boudec(alpha: float, times: List[float]) -> Tuple[float, float]:
    sorted_times = sorted(times)
    n = len(times)

    z_value = {0.95: 1.96, 0.99: 2.576}.get(alpha)
    assert z_value

    low_pos = math.floor((n - z_value * math.sqrt(n)) / 2)
    high_pos = math.ceil(1 + (n + z_value * math.sqrt(n)) / 2)

    return (sorted_times[low_pos], sorted_times[high_pos])    

def compute_statistics(df):
    times = df['times']
    mean = round(df['times'].mean(),3)
    variation = round(scipy.stats.variation(df['times']),3)
    std_dev = round(np.std(df['times']),3)
    median = round(df['times'].median(),3)
    
    for alpha in [0.95]:
    #for alpha in [0.95, 0.99]:
      #parametric
      ci_interval = scipy.stats.t.interval(alpha, len(times) - 1, loc=mean, scale=st.sem(times))
      interval_width = ci_interval[1] - ci_interval[0]
      ratio = 100 * interval_width / mean / 2.0
    
      #non-parametric
      ci_interval = ci_le_boudec(alpha, times)
      interval_width = ci_interval[1] - ci_interval[0]
      ratio = 100 * interval_width / median / 2.0
      
      #round values
      ci_interval_0 = round(ci_interval[0],2)
      ci_interval_1 = round(ci_interval[1],2)      
      ratio = round(ratio,2)
      
      if ratio > 5.0:
        print(f"Non-parametric CI {alpha} from {ci_interval_0} to {ci_interval_1}, within {ratio}% of median")

   

def randomcolor(data):
    x = hash(repr(data))
    r = 25 + x % 100
    g = 25 + int(x / 0xfa) % 100
    b = 50 + int(x / 0xff) % 100
    return "#" + hex(r)[2:].rjust(2,'0') + hex(g)[2:].rjust(2,'0') + hex(b)[2:].rjust(2,'0')

def compute_total(df):
    dfs = []
    all_req_ids = []

      
    req_ids = df["request_id"].unique()[:180]
    df_i = df.loc[df["request_id"].isin(req_ids)]
    
    disjoint = set(req_ids).isdisjoint(all_req_ids)
    all_req_ids.extend(req_ids)
                                
    invos = df_i.groupby("request_id")
    d_runtime = invos["end"].max() - invos["start"].min()
    new_df = pd.DataFrame(d_runtime, columns=["times"])
    
    dfs.append(new_df)
    
    compute_statistics(pd.concat(dfs))

def compute_per_repetition(df, config):
    dfs = []
    all_req_ids = []

    for i in range(0,6,1):
      print("Repetition ", i)
    
      start = i * 30
      end = start + 30
      
      req_ids = df["request_id"].unique()[start:end]
      df_i = df.loc[df["request_id"].isin(req_ids)]
      
      all_req_ids.extend(req_ids)
                                  
      invos = df_i.groupby("request_id")
      d_runtime = invos["end"].max() - invos["start"].min()
      new_df = pd.DataFrame(d_runtime, columns=["times"])
      
      coeff_variation = str(round(scipy.stats.variation(df.loc[i:i+300]['times']),2))
      
      exp_id = ""
      if config == '2022':
        exp_id += "2022\n Repetition "
      else:
        exp_id = "2024\n Repetition " 
        
      exp_id += str(i) + "\nVariation " + coeff_variation + "\navg runtime: " + str(round(new_df["times"].mean(),2))
        
      new_df['exp_id'] = exp_id
      
      dfs.append(new_df)
      
      compute_statistics(pd.concat(dfs))
    return dfs



def violin_plot():
    sb.set_theme()
    sb.set_context("paper")
    ax = plt.gca()

    benchmarks = [args.benchmark]
    if not benchmarks[0]:
        benchmarks = [p for p in os.listdir("./../perf-cost") if os.path.isdir(os.path.join("./../perf-cost", p))]


    dfs = []
    for benchmark in benchmarks:
        for platform in args.platforms:
            configs = args.config
            if len(configs) == 0:
                configs = ["../" + platform]
                
                print(configs)
    
            for experiment in args.experiments:
                for memory in args.memory:
                    #for config in configs:
                    for size in ['2e5', '2e8', '2e10', '2e12', '2e14', '2e16', '2e18-1000']:
                        #for duration in ['1', '5', '10', '15', '20']:
                    
                        filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                        
                        path = os.path.join("./../perf-cost", benchmark, platform + "_" + size, filename)
                        if not os.path.exists(path):
                            print(path)
                            continue

                        if platform == "azure":
                          df = read(path)
                        else:
                          df = pd.read_csv(path)
                        
                        df["exp_id"] = ""
                        df["times"] = df["end"] - df["start"]
                        
                        print(memory, size)
                        compute_total(df) 

        ax.set_ylabel("Duration [s]")
        ax.set_xlabel(None)
        plt.title(benchmarks[0] + " on " + args.platforms[0])
        plt.tight_layout()
        
if __name__ == "__main__":
    
    violin_plot()
