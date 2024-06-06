import pandas as pd
import scipy
import os
import argparse
import json

from typing import List, Tuple
from collections import namedtuple

#import seaborn as sb
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", type=str)
parser.add_argument("-m", "--memory", type=str)
parser.add_argument("-p", "--platform", type=str)
parser.add_argument("-e", "--experiment", type=str)
args = parser.parse_args()

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
    
    for alpha in [0.95, 0.99]:
      #parametric
      ci_interval = scipy.stats.t.interval(alpha, len(times) - 1, loc=mean, scale=st.sem(times))
      interval_width = ci_interval[1] - ci_interval[0]
      ratio = 100 * interval_width / mean / 2.0
      print(f"Parametric CI (Student's t-distribution) {alpha} from {ci_interval[0]} to {ci_interval[1]}, within {ratio}% of mean")
    
      #non-parametric
      ci_interval = ci_le_boudec(alpha, times)
      interval_width = ci_interval[1] - ci_interval[0]
      ratio = 100 * interval_width / median / 2.0
      
      #round values
      ci_interval_0 = round(ci_interval[0],2)
      ci_interval_1 = round(ci_interval[1],2)      
      ratio = round(ratio,2)
      
      #print(f"Non-parametric CI {alpha} from {ci_interval_0} to {ci_interval_1}, within {ratio}% of median")


def plot_cold_vs_warm(df, cold_entries, warm_entries):
    dfs = []
    sort_after = 'is_cold'
    
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
    funcs = cold_entries.groupby("func")
    for key, item in funcs:
      my_df = funcs.get_group(key)["times"]
      print("func: ", key, my_df.mean(), "length: ", len(my_df))
      #now write to new df
      new_df = pd.DataFrame(my_df, columns=["times"])
      exp_id = ""
      exp_id += "func: " + key + "\n" + sort_after + "True" + "\nmean " + str(round(new_df['times'].mean(),2))
      new_df['exp_id'] = exp_id

      dfs.append(new_df)
      
    funcs = warm_entries.groupby("func")
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

    #sb.violinplot(x="exp_id", y="times", data=df, cut=0) 

    ax.set_ylabel("Duration [s]")
    ax.set_xlabel(None)
    #plt.show()

def randomcolor(data):
    x = hash(repr(data))
    r = 25 + x % 100
    g = 25 + int(x / 0xfa) % 100
    b = 50 + int(x / 0xff) % 100
    return "#" + hex(r)[2:].rjust(2,'0') + hex(g)[2:].rjust(2,'0') + hex(b)[2:].rjust(2,'0')
    #print(len(colors))
    #return colors[x % (len(colors) - 1)]

#sb.set_theme()
#sb.set_context("paper")
#ax = plt.gca()

memory = args.memory
sizes = ['2e5', '2e8', '2e10', '2e12', '2e14', '2e16', '2e18-1000']

for size in sizes:
    #batch-size-30-reps-6
    
    #benchmark_path = os.path.join("../perf-cost", args.benchmark, args.platform, "batch-size-30-reps-6", args.experiment + "_" + memory + "_processed.csv")
    
    benchmark_path = os.path.join("../perf-cost", args.benchmark, args.platform + "_" + size, args.experiment + "_" + memory + ".csv")
    #benchmark_path = os.path.join("../perf-cost", args.benchmark, args.platform, args.experiment + "_" + memory + "_processed.csv")
    
    #benchmark_path = os.path.join("../perf-cost", args.benchmark, args.platform + "_" + size, args.experiment + ".csv")
    #benchmark_path = os.path.join("../perf-cost", args.benchmark, args.platform + "_" + size, args.experiment + "_processed.csv")
    df = pd.read_csv(benchmark_path)
    df = df[df["func"] != "run_workflow"]


    df['times'] = df['end'] - df['start']

    df_req = df.groupby("request_id")
    for key, item in df_req:
        my_df = df_req.get_group(key)
        #print(my_df['is_cold'])
        #was_cold_start = not any(df['is_cold'] == False)
        my_df = my_df.sort_values("start").reset_index(drop=True) #["is_cold"] #.at[0, "is_cold"]
        was_cold_start = my_df.at[1, "is_cold"]
        if was_cold_start: 
            print("was_cold_start: ", was_cold_start)

    #print(df['is_cold'])
    cold_entries = df.loc[df['is_cold'] == True]
    warm_entries = df.loc[df['is_cold'] == False]

    #how many requests are completely warm? find out normal size of one request (groupby
    len_one_request_id = df.groupby("request_id").size().min()
    print(len_one_request_id, "workflow executions total: ", len(df.groupby("request_id")))
    warm_reqs = warm_entries.groupby("request_id").size()
    cold_reqs = cold_entries.groupby("request_id").size()
    #every warm_entry_by_request whose length is 4.
    warm_invocs = warm_reqs.loc[warm_reqs == len_one_request_id]
    cold_invocs = cold_reqs.loc[cold_reqs == len_one_request_id]

    #print(warm_invocs)

    #compute_statistics(warm_invocs)

    print("warm invocs: ", len(warm_invocs))
    print("cold invocs: ", len(cold_invocs))
    print("warm requests functions: ", warm_entries.groupby("func").size())

    print("#cold entries: ", len(cold_entries), "#warm_entries: ", len(warm_entries), "total entries: ", len(df))



