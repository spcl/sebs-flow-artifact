import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
parser.add_argument("-v", "--visualization", choices=["bar", "violin", "line"], default="bar")
parser.add_argument("--sum", action="store_true", default=False)
parser.add_argument("--all", action="store_true", default=False)
args = parser.parse_args()

platform_names = {
    "aws/2022": "AWS",
    "azure/2022": "Azure",
    "gcp/2022": "Google Cloud",
    "aws/2024": "AWS",
    "gcp/2024": "Google Cloud",
    "azure/2024": "Azure"
}

colors = sb.color_palette("colorblind")
color_map = {
    "AWS": colors[0],
    'aws/2024': colors[0], #'#0173b2', 
    "AWS Docs": colors[3],
    "Azure": colors[1],
    'azure/2024': colors[1], #'#de8f05', 
    "Google Cloud": colors[2],
    'gcp/2024': colors[2], #'#029e73', 
    "Google Cloud Docs": colors[4],
    "AWS new": colors[5],
    "GCP new": colors[6],
    "AWS new 2": colors[6]
}

def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:180]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    if np.any(df["duration"] <= 0):
        raise ValueError(f"Invalid data frame at path {path}")

    return df


def bar_plot():
    sb.set_theme()
    sb.set_context("paper")
    colors = sb.color_palette("colorblind")
    fig, ax = plt.subplots()
    w = 0.3
    xs = np.arange(len(args.experiments))

    for idx, platform in enumerate(args.platforms):
        ys = []
        es = []

        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                #filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}_processed.csv"
                #path = os.path.join("./../perf-cost", args.benchmark, platform + "_2e10", filename)
                path = os.path.join("./../perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    ys.append(0)
                    es.append(0)
                    print(path)
                    continue

                df = read(path)
                invos = df.groupby("request_id")

                if args.sum:
                    d_total = invos["duration"].sum()
                else:
                    d_total = invos["end"].max() - invos["start"].min()
                    print("total: ", d_total)
                y = np.mean(d_total)
                e = st.t.interval(confidence=0.95, df=len(d_total)-1, loc=y, scale=st.sem(d_total))
                e = np.abs(e-y)

                ys.append(np.mean(d_total))
                es.append(np.mean(e))

        name = platform_names[platform]
        
        o = 0
        if "gcp" in platform:
            at = .5
        elif "aws" in platform:
            at = 1
        else:
            at = 1.5
        #o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        name = platform_names[platform]
        color = color_map[name]
        #bar = ax.bar(at, ys, w, yerr=cse, label=name, color=color, capsize=3, linewidth=1)
        
        #o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        print("es = ", es)
        #ax.bar(xs-o, ys, w, label=name, yerr=es, capsize=3, color=color_map[name])
        box = ax.boxplot(at, ys, w)
        color = color_map[name]
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    ax.set_ylabel("Duration [s]")
    ax.set_xticks(xs, args.experiments)
    fig.legend(bbox_to_anchor=(0.97, 0.97))

    plt.tight_layout()
    plt.show()


def violin_plot():
    sb.set_theme()
    sb.set_context("paper")
    #colors = sb.color_palette("colorblind")
    benchmarks = [args.benchmark]
    if not benchmarks[0]:
        benchmarks = [p for p in os.listdir("./../perf-cost") if os.path.isdir(os.path.join("./../perf-cost", p))]

    w = .2
    dfs = []
    for benchmark in benchmarks:
        for platform in args.platforms:
            for experiment in args.experiments:
                for memory in args.memory:
                    filename = f"{experiment}_{memory}.csv" if platform != "azure/2022" else f"{experiment}_processed.csv"
                    #path = os.path.join("./../perf-cost", args.benchmark, platform + "_2e10", filename)
                    #filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                    filename = f"{experiment}_{memory}_processed.csv"
                    path = os.path.join("./../perf-cost", benchmark, platform, filename)
                    if not os.path.exists(path):
                        print(path)
                        continue

                    meta = {"platform": platform, "experiment": experiment, "memory": memory, "benchmark": benchmark}
                    df = read(path)
                    invos = df.groupby("request_id")

                    d_total = invos["duration"].sum()
                    
                    d_runtime = invos["end"].max() - invos["start"].min()
                    
                    data = d_total if args.sum else d_runtime

                    cold_starts = invos["is_cold"].sum()

                    print(platform.split("/")[0], "median runtime:", np.median(data))

                    #print(platform, experiment, "avg runtime:", np.mean(data), "avg num cold starts:", np.mean(cold_starts))
                    df = pd.DataFrame(data, columns=["duration"])
                    df = df.assign(**meta)
                    df["exp_id"] = df["platform"] #+"\n"+df["experiment"]+"\n"+df["memory"]

                    

                    dfs.append(df)

        df = pd.concat(dfs)

        ax = plt.gca()
        
        box_plot = sb.boxplot(x="exp_id", y="duration", data=df, palette=color_map)
        #elif "aws" in platform:
            
        #box_plot = sb.boxplot(x="exp_id", y="duration", data=df, palette=color_map, boxprops=dict(color='blue'))
        
        #for patch in box_plot.artists:
        #    print("patch")
        #    fc = patch.get_facecolor()
        #    patch.set_facecolor('blue')

        medians = df.groupby(['exp_id'])['duration'].median()
        medians = medians.round(2)
        vertical_offset = df['duration'].median() * 0.5# offset from median for display

        ax.set_xticklabels((platform_names[p] for p in args.platforms), fontsize=16)
        for xtick in box_plot.get_xticks():
            #print("xtick: ", xtick, "medians: ", medians[xtick])
            if xtick == 1:
                #AWS
                box_plot.text(xtick,medians[(xtick-1)%3] + vertical_offset +20 ,medians[(xtick-1)%3], 
                    horizontalalignment='center',size='20')#, weight='semibold')
            elif xtick == 2:
                #Azure
                box_plot.text(xtick,medians[(xtick-1)%3] + vertical_offset +.01,medians[(xtick-1)%3], 
                    horizontalalignment='center',size='20')#, weight='semibold')
            else:
                #GCP
                box_plot.text(xtick,medians[(xtick-1)%3] + vertical_offset+25,medians[(xtick-1)%3], 
                    horizontalalignment='center',size='20')#, weight='semibold')


        
        #ax.set_xticklabels(platform_names[p.split("/")[0]] for p in args.platforms)
        
        ax.set_ylabel("Duration [s]", fontsize=16)
        ax.set_xlabel(None)
        #ax.set_yscale(fontsize=16)
        plt.yticks(fontsize=16)
        
        #ticks for video analysis
        #ticks = [25, 50, 100, 250, 500, 750, 1000]
        #ticks for exCamera
        #ticks = [75, 100, 250, 500, 750, 1000, 1500]
        #ax.set_yticks(ticks)
        #ax.set_yticklabels(ticks)

        plt.tight_layout()
        plt.savefig("../figures/plots/runtime/runtime-" + args.benchmark + ".pdf")
        
        #plt.show()


def line_plot():
    fig, ax = plt.subplots()
    x_axis_len = []

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                path = os.path.join("./../perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    continue

                df = read(path)
                invos = df.groupby("request_id")
                if args.sum:
                    d_total = invos["duration"].sum()
                else:
                    d_total = invos["end"].max() - invos["start"].min()

                ys = np.asarray(d_total)
                xs = np.arange(ys.shape[0])

                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")

                ys = ys[np.where(~np.isnan(ys))]
                print(platform, "std:", np.std(ys))
                x_axis_len.append(len(xs))

    # ax.set_title(f"{args.benchmark} Runtime")
    ax.set_xlabel("Repetition")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, min(x_axis_len)+1, 5))
    
    ax.set_ylabel("Duration [s]")
    ax.set_xlim([0, min(x_axis_len)-1])
    fig.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plots = {"bar": bar_plot, "violin": violin_plot, "line": line_plot}
    plot_func = plots[args.visualization]

    if args.all:
        benchmarks = ["650.vid", "660.map-reduce", "670.auth", "680.excamera", "690.ml"]
        experiments = ["burst", "warm", "cold"]
        memory = ["128", "256", "1024", "2048"]
        for b in benchmarks:
            for e in experiments:
                for m in memory:
                    args = Bunch(benchmark=b, experiments=[e], memory=[m], platforms=["aws", "azure", "gcp"], sum=False)

                    path = os.path.join("/Users/2022/Desktop", "res", "runtime", f"{b}-{e}-{m}.pdf")
                    try:
                        plot_func(path)
                    except:
                        pass
                    else:
                        print(path)
    else:
        plot_func()
