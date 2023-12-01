import os
import argparse
import json
import dateutil.parser

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", type=str)
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-m", "--memory", nargs='+', default=[])
args = parser.parse_args()


platform_names = {
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud"
}

colors = sb.color_palette("colorblind")
color_map = {
    "AWS": colors[0],
    "AWS Docs": colors[3],
    "Azure": colors[1],
    "Google Cloud": colors[2],
    "Google Cloud Docs": colors[4],
}

def randomcolor(data):
    x = hash(repr(data))
    r = 25 + x % 100
    g = 25 + int(x / 0xff) % 100
    b = 25 + int(x / 0xffff) % 100
    return "#" + hex(r)[2:].rjust(2,'0') + hex(g)[2:].rjust(2,'0') + hex(b)[2:].rjust(2,'0')
    #print(len(colors))
    #return colors[x % (len(colors) - 1)]

def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:30]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


if __name__ == "__main__":
    sb.set_theme(style="whitegrid")
    #sb.set_context("paper")
    fig, ax = plt.subplots()

    azure_experiments = set()

    for idx, platform in enumerate(args.platforms):
        for experiment in args.experiments:
            for memory in args.memory:
                if platform == "azure":
                    if experiment in azure_experiments:
                        continue
                    else:
                        azure_experiments.add(experiment)

                
                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                #path = os.path.join("./../perf-cost", args.benchmark, platform, filename)
                path = os.path.join("./../perf-cost", args.benchmark, platform+"_2e27", filename)
                #path = os.path.join("./../perf-cost", args.benchmark, platform+"_vpc_10", filename)
                #path = os.path.join("./../perf-cost", args.benchmark, platform+"_4t_20s", filename)
                if not os.path.exists(path):
                    print(path)
                    continue

                df = read(path)
                start = df["start"].min()
                df["start_diff"] = df["start"] - start
                invos = df.groupby("request_id")
                req_ids = invos.agg({"start_diff": min}).sort_values("start_diff").head(30).index
                df = df[df["request_id"].isin(req_ids)] # only select the first 30 invos

                req_ids = df["request_id"].unique()
                assert(len(req_ids) == 30)

                # check if they have all been invoked simultaneously
                if experiment == "burst":
                    filename = f"{experiment}_results_{memory}.json" if platform != "azure" else f"{experiment}_results.json"
                    #path = os.path.join("./../perf-cost", args.benchmark, platform, filename)
                    path = os.path.join("./../perf-cost", args.benchmark, platform+"_2e27", filename)
                    #path = os.path.join("./../perf-cost", args.benchmark, platform+"_4t_20s", filename)
                    with open(path) as f:
                        results = json.load(f)
                    invos = results["_invocations"]
                    invos = invos[list(invos.keys())[0]]
                    starts = []
                    for req_id in req_ids:
                        client_begin = invos[req_id]["times"]["client_begin"]
                        client_begin = dateutil.parser.parse(client_begin).timestamp()
                        starts.append(client_begin)

                    diff = max(starts) - min(starts)
                    print(platform, experiment, diff)
                    #assert(diff < 1)
                exp_start = df["start"].min()

                #let everything start at 0
                df["start"] -= exp_start
                df["end"] -= exp_start

                task_colors = {'analyse': 'c', 'summarize': 'm', 'decode': 'y'}

                #print(df)
                print("number of containers: ", df['container_id'].unique().size)
                print("number of requests: ", df['request_id'].unique().size)
                download_funcs = df.loc[df['func'] == 'process']
                print("mean duration: ", download_funcs['duration'].mean())

                #get total number of containers

                for index, row in df.iterrows():
                    #container_color = 
                    #plt.barh(y=row['request_id'], width=row['duration'], left=row['start'], color=task_colors[row['func']], alpha=0.7)
                    #print(row['container_id'])
                    plt.barh(y=row['request_id'], width=row['duration'], left=row['start'], color=randomcolor([row['container_id']]), alpha=0.7)
                    #plt.barh(y=row['container_id'], width=row['duration'], left=row['start'], color=randomcolor([row['container_id']]), alpha=0.7)
                plt.tight_layout()
                #plt.savefig("/home/larissa/Paper/SIGMETRICS-WorkflowsBenchmarks/figures/timeline/color-per-container/timeline-" + args.benchmark + "-m" + memory + "-" + experiment + "-" + platform + ".pdf")
                plt.show()

                #for index, row in df.iterrows():
                #    plt.barh(y=row['task'], width=row['task_duration'], left=row['days_to_start'] + 1, color=team_colors[row['team']])


                '''
                ts = np.sort(pd.concat([df["start"], df["end"]]).unique())

                xs = []
                ys = []
                cs = set()
                _t = -1
                for t in ts:
                    assert(t > _t)
                    _t = t

                    cs_start = df.loc[(df["start"] == t)]["container_id"]
                    cs.update(cs_start)

                    cs_end = df.loc[(df["end"] == t)]["container_id"]
                    cs = cs.difference(cs_end)

                    assert(len(cs_start) + len(cs_end) > 0)

                    xs.append(t)
                    ys.append(len(cs))

                xs = np.asarray(xs)
                ys = np.asarray(ys)

                #print(platform, "duration:", df["end"].max(), "max threads:", max(ys))
                #name = platform_names[platform]
                #line = ax.plot(xs, ys, zorder=len(platform)-idx, color=color_map[name])[0]
                #line.set_label(name)

                #if platform == "gcp":
                #    ts = [47, 86.5]
                #    ts = [xs[np.argmin(np.abs(xs - t))] for t in ts]

                #    ax.fill_between(xs, ys, where=(xs <= ts[0]), alpha=0.2, color=line.get_color(), linewidth=0.0, label="encode")
                #    ax.fill_between(xs, ys, where=((xs >= ts[0]) & (xs <= ts[1])), alpha=0.5, color=line.get_color(), linewidth=0.0, label="reencode")
                #    ax.fill_between(xs, ys, where=(xs >= ts[1]), alpha=0.7, color=line.get_color(), linewidth=0.0, label="rebase")
'''

    # ax.set_title(f"{args.benchmark} scalability")
    #ax.set_xlabel("Time [s]")
    #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #ax.set_ylabel("#Threads")
    #fig.legend(bbox_to_anchor=(0.97, 0.97))
    # ax.set_xscale("log")

    #plt.tight_layout()
    #plt.savefig("/home/larissa/Paper/SIGMETRICS-WorkflowsBenchmarks/figures/vis/timeline-" + args.benchmark + "-mem-" + args.memory[0] + ".pdf")
    #plt.show()
