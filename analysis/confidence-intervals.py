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
args = parser.parse_args()


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

    # z(alfa/2)
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

    print(f'Median {median} mean {mean} std_dev {std_dev} variation {variation}')    
    
    for alpha in [0.95]:
    #for alpha in [0.95, 0.99]:
      #parametric
      ci_interval = scipy.stats.t.interval(alpha, len(times) - 1, loc=mean, scale=st.sem(times))
      interval_width = ci_interval[1] - ci_interval[0]
      ratio = 100 * interval_width / mean / 2.0
      #print(f"Parametric CI (Student's t-distribution) {alpha} from {ci_interval[0]} to {ci_interval[1]}, within {ratio}% of mean")
    
      #non-parametric
      ci_interval = ci_le_boudec(alpha, times)
      interval_width = ci_interval[1] - ci_interval[0]
      ratio = 100 * interval_width / median / 2.0
      
      #round values
      ci_interval_0 = round(ci_interval[0],2)
      ci_interval_1 = round(ci_interval[1],2)      
      ratio = round(ratio,2)
      
      print(f"Non-parametric CI {alpha} from {ci_interval_0} to {ci_interval_1}, within {ratio}% of median")

   

def randomcolor(data):
    x = hash(repr(data))
    r = 25 + x % 100
    g = 25 + int(x / 0xfa) % 100
    b = 50 + int(x / 0xff) % 100
    return "#" + hex(r)[2:].rjust(2,'0') + hex(g)[2:].rjust(2,'0') + hex(b)[2:].rjust(2,'0')

def violin_plot():
    sb.set_theme()
    sb.set_context("paper")
    ax = plt.gca()

    benchmarks = [args.benchmark]
    if not benchmarks[0]:
        benchmarks = [p for p in os.listdir("./../perf-cost") if os.path.isdir(os.path.join("./../perf-cost", p))]


    dfs = []
    all_req_ids = []
    for benchmark in benchmarks:
        for platform in args.platforms:    
            for experiment in args.experiments:
                for memory in args.memory:
                    filename = f"{experiment}_{memory}_processed.csv" 
                   
                    path = os.path.join("./../perf-cost", benchmark, platform, filename)
                    if not os.path.exists(path):
                        print("does not exist: ", path)
                        continue

                    if platform == "azure/2024":
                      df = read(path)
                    else:
                      df = pd.read_csv(path)
                    print(path)
                    print(len(df))
                    #df = df.sort_values('start')
                    
                    df["exp_id"] = ""
                    df["times"] = df["end"] - df["start"]
                                            
                    for i in range(0,6,1):
                      print("Repetition ", i)
                      
                      #each repetition has 30 invocations. 
                      start = i * 30
                      end = start + 30
                      
                      req_ids = df["request_id"].unique()[start:end]
                      df_i = df.loc[df["request_id"].isin(req_ids)]
                      
                      disjoint = set(req_ids).isdisjoint(all_req_ids)
                      all_req_ids.extend(req_ids)
                                                  
                      invos = df_i.groupby("request_id")
                      d_runtime = invos["end"].max() - invos["start"].min()
                      new_df = pd.DataFrame(d_runtime, columns=["times"])
                      
                      coeff_variation = str(round(scipy.stats.variation(df.loc[i:i+300]['times']),2))
                      
                      
                      new_df['exp_id'] = 'Repetition ' + str(i)
                      dfs.append(new_df)
                      
                      compute_statistics(pd.concat(dfs))


        df = pd.concat(dfs)
        df = df.sort_values('exp_id')
        my_pal = {exp_id: randomcolor(exp_id.split('\n')[0]) for exp_id in df["exp_id"].unique()}
        sb.violinplot(x="exp_id", y="times", data=df, cut=0, palette=my_pal)
                    
        ax.set_ylabel("Duration [s]")
        ax.set_xlabel(None)
        
        plt.title(benchmarks[0] + " on " + args.platforms[0])
        plt.tight_layout()
        
        plt.show()


if __name__ == "__main__":
    
    violin_plot()
