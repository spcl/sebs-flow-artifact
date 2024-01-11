import os
import argparse

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
    "aws-new": "AWS new",
    "gcp-new": "GCP new",
    "aws-new-2": "AWS new 2",
    "gcp-32invocs-6-1": "gcp-32invocs-6-1",
    "gcp-32invocs-6-1-first": "gcp-32invocs-6-1-first",
    "gcp-96invocs-7-1": "gcp-96invocs-7-1",
    "gcp-96invocs-7-1-first": "gcp-96invocs-7-1-first",
    "aws-32invocs-8-1": "aws-32-invocs-8-1",
    "aws-32invocs-8-1-first": "aws-32-invocs-8-1-first",
    "aws-96invocs-7-1": "aws-96invocs-7-1",
    "aws-96invocs-7-1-first": "aws-96invocs-7-1-first"
}

colors = sb.color_palette("colorblind")
color_map = {
    "AWS": colors[0],
    "AWS Docs": colors[3],
    "Azure": colors[1],
    "Google Cloud": colors[2],
    "Google Cloud Docs": colors[4],
    "AWS new": colors[5],
    "GCP new": colors[6],
    "AWS new 2": colors[6]
}

def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:30]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    if np.any(df["duration"] <= 0):
        raise ValueError(f"Invalid data frame at path {path}")

    return df

def randomcolor(data):
    x = hash(repr(data))
    r = 25 + x % 100
    g = 25 + int(x / 0xfa) % 100
    b = 50 + int(x / 0xff) % 100
    return "#" + hex(r)[2:].rjust(2,'0') + hex(g)[2:].rjust(2,'0') + hex(b)[2:].rjust(2,'0')
    #print(len(colors))
    #return colors[x % (len(colors) - 1)]

def violin_plot():
    sb.set_theme()
    sb.set_context("paper")
    ax = plt.gca()

    #sb.color_palette("colorblind")
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
                    for config in configs:
                        filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                        #filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                        
                        path = os.path.join("./../perf-cost", benchmark, platform, config, filename)
                        if not os.path.exists(path):
                            print(path)
                            continue

                        df = pd.read_csv(path)
                        #df1 = df.drop(df[df['func'] != 'individuals'].index)
                        #df2 = df.drop(df[df['func'] != 'frequency'].index)
                        #df = pd.concat([df1, df2])
                        
                        df["times"] = df["end"] - df["start"]
                        #df["exp_id"] = df["end"] - df["start"]
                        funcs = df.groupby("func")
                        for key, item in funcs:
                            my_df = funcs.get_group(key)["times"]
                            coeff_variation = str(round(scipy.stats.variation(my_df),2))
                            
                            print_config=config.replace("-","\n")
                            
                            exp_id = df["func"] + "\n"
                            if config == 'laurin':
                              exp_id += "2022\n"
                            elif config == 'batch-size-32-reps-3':
                              exp_id += "2023, 7.1.\n"
                            else:
                              exp_id += "2023, 9.1.\n" 
                            exp_id += "variation: " + coeff_variation
                            exp_id += "\navg runtime: " + str(round(my_df.mean(),2))
                            df.loc[df['func'] == key, 'exp_id'] = exp_id


                        dfs.append(df)
                        print(df)
        df = pd.concat(dfs)
        df = df.sort_values('exp_id')
        my_pal = {exp_id: randomcolor(exp_id.split('\n')[0]) for exp_id in df["exp_id"].unique()}
        sb.violinplot(x="exp_id", y="times", data=df, cut=0, palette=my_pal)

        ax.set_ylabel("Duration [s]")
        ax.set_xlabel(None)

        plt.title(benchmarks[0] + " on " + args.platforms[0])
        plt.tight_layout()
        
        plt.savefig("/home/larissa/Serverless/benchmark-workflows/meetings/" + benchmark + "-per-function-aws-gcp.pdf")
        
        plt.show()


if __name__ == "__main__":
    
    violin_plot()
