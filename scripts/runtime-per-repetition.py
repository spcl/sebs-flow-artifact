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
#                        filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                        
                        path = os.path.join("./../perf-cost", benchmark, platform, config, filename)
                        print(path)
                        if not os.path.exists(path):
                            print(path)
                            continue

                        df = pd.read_csv(path)
                        print(path)
                        print(len(df))
                        df.sort_values('start')
                        df["exp_id"] = ""
                        df["times"] = df["end"] - df["start"]
                        
                        if config == 'batch-size-32-reps-3': 
                          for i in range(0,len(df)+1,300):
                            df_i = df.loc[i:i+299]
                            invos = df_i.groupby("request_id")

                            d_runtime = invos["end"].max() - invos["start"].min()
                            new_df = pd.DataFrame(d_runtime, columns=["times"])
                            
                            coeff_variation = str(round(scipy.stats.variation(df.loc[i:i+300]['times']),2))
                            
                            exp_id = ""
                            if config == 'laurin':
                              exp_id += "2022\n Repetition "
                              print_exp_id = "2023 rep " + str(int(i/300)+1)
                            elif config == 'batch-size-32-reps-3':
                              exp_id = "2023, 7.1.\n Repetition "
                              print_exp_id = "2023 rep " + str(int(i/300)+1)
                            else:
                              exp_id = "2023, 9.1.\n Repetition " 
                              print_exp_id = "2023 rep " + str(int(i/300)+1)
                              
                            exp_id += str(int(i/300)+1) + "\nVariation " + coeff_variation + "\navg runtime: " + str(round(new_df["times"].mean(),2))
                              
                            new_df['exp_id'] = exp_id
                            print("exp_id = ", print_exp_id, "avg time\n", str(round(new_df["times"].mean(),2)))
                            dfs.append(new_df)
                        else: 
                          #my config + laurin has 30 invocations. 
                          for i in range(0,len(df)+1,300):
                            df_i = df.loc[i:i+299]
                            invos = df_i.groupby("request_id")

                            d_runtime = invos["end"].max() - invos["start"].min()
                            new_df = pd.DataFrame(d_runtime, columns=["times"])
                            
                            coeff_variation = str(round(scipy.stats.variation(df.loc[i:i+300]['times']),2))
                            
                            exp_id = ""
                            if config == 'laurin':
                              exp_id += "2022\n Repetition "
                              print_exp_id = "2023 rep " + str(int(i/300)+1)
                            elif config == 'batch-size-32-reps-3':
                              exp_id = "2023, 7.1.\n Repetition "
                              print_exp_id = "2023 rep " + str(int(i/300)+1)
                            else:
                              exp_id = "2023, 9.1.\n Repetition " 
                              print_exp_id = "2023 rep " + str(int(i/300)+1)
                              
                            exp_id += str(int(i/300)+1) + "\nVariation " + coeff_variation + "\navg runtime: " + str(round(new_df["times"].mean(),2))
                              
                            new_df['exp_id'] = exp_id
                            print("exp_id = ", print_exp_id, "avg time\n", str(round(new_df["times"].mean(),2)))
                            dfs.append(new_df)


        df = pd.concat(dfs)
        df = df.sort_values('exp_id')
        #print(df)
        my_pal = {exp_id: randomcolor(exp_id.split('\n')[0]) for exp_id in df["exp_id"].unique()}
        sb.violinplot(x="exp_id", y="times", data=df, cut=0, palette=my_pal)
                    
        '''
                    #group by function
                    df["times"] = df["end"] - df["start"]
                    df["exp_id"] = df["func"]+"\n"+platform+"\n"+experiment+"\n"+memory
                    print("exp_id", df["exp_id"])
                    
                    invos = df.groupby("func")

                    for key, item in invos:
                      print(key)
                      my_df = invos.get_group(key) #["times"]
                      #variation = scipy.stats.variation(my_df)
                      #meta = {"platform": platform, "experiment": experiment, "memory": memory, "benchmark": benchmark}
                      #my_df = my_df.assign(**meta)

                      sb.violinplot(x="exp_id", y="times", data=my_df, cut=0)
                      #print(platform, experiment, "avg runtime:", np.mean(data), "avg num cold starts:", np.mean(cold_starts))

                    dfs.append(df)
        '''

        #df = pd.concat(dfs)

        #fig, ax = plt.subplots()
        #sb.violinplot(x="exp_id", y="duration", data=df, cut=0)
        #sb.boxplot(x="exp_id", y="duration", data=df)
        #ax.boxplot(x="exp_id", y="duration", data=df)
        #ax.set_xticklabels(platform_names[p] for p in args.platforms)

        ax.set_ylabel("Duration [s]")
        ax.set_xlabel(None)
        #ax.set_yscale("log")
        
        #ticks for video analysis
        #ticks = [25, 50, 100, 250, 500, 750, 1000]
        #ticks for exCamera
        #ticks = [75, 100, 250, 500, 750, 1000, 1500]
        #ax.set_yticks(ticks)
        #ax.set_yticklabels(ticks)
        plt.title(benchmarks[0] + " on " + args.platforms[0])
        plt.tight_layout()
        
        #plt.savefig("/home/larissa/Serverless/benchmark-workflows/meetings/" + benchmark + "-per-function-aws-gcp.pdf")
        
        plt.show()


if __name__ == "__main__":
    
    violin_plot()
