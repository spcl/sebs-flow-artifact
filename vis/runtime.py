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
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud"
}

colors = sb.color_palette()
color_map = {
    "AWS": colors[0],
    "AWS Docs": colors[3],
    "Azure": colors[1],
    "Google Cloud": colors[2],
    "Google Cloud Docs": colors[4],
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


def bar_plot():
    fig, ax = plt.subplots()
    w = 0.3
    xs = np.arange(len(args.experiments))

    for idx, platform in enumerate(args.platforms):
        ys = []
        es = []

        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    ys.append(0)
                    es.append(0)
                    continue

                df = read(path)
                invos = df.groupby("request_id")

                if args.sum:
                    d_total = invos["duration"].sum()
                else:
                    d_total = invos["end"].max() - invos["start"].min()

                y = np.mean(d_total)
                e = st.t.interval(alpha=0.95, df=len(d_total)-1, loc=y, scale=st.sem(d_total))
                e = np.abs(e-y)

                ys.append(np.mean(d_total))
                es.append(e)

        name = platform_names[platform]
        o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        ax.bar(xs-o, ys, w, label=name, yerr=es, capsize=3, color=color_map[name])

    ax.set_ylabel("Duration [s]")
    ax.set_xticks(xs, args.experiments)
    fig.legend(bbox_to_anchor=(0.97, 0.97))

    plt.tight_layout()
    plt.show()


def violin_plot():
    benchmarks = [args.benchmark]
    if not benchmarks[0]:
        benchmarks = [p for p in os.listdir("perf-cost") if os.path.isdir(os.path.join("perf-cost", p))]

    dfs = []
    for benchmark in benchmarks:
        for platform in args.platforms:
            for experiment in args.experiments:
                for memory in args.memory:
                    filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                    path = os.path.join("perf-cost", benchmark, platform, filename)
                    if not os.path.exists(path):
                        continue

                    meta = {"platform": platform, "experiment": experiment, "memory": memory, "benchmark": benchmark}
                    df = read(path)
                    invos = df.groupby("request_id")

                    d_total = invos["duration"].sum()
                    d_runtime = invos["end"].max() - invos["start"].min()
                    data = d_total if args.sum else d_runtime

                    cold_starts = invos["is_cold"].sum()

                    print(platform, experiment, "avg runtime:", np.mean(data), "avg num cold starts:", np.mean(cold_starts))

                    df = pd.DataFrame(data, columns=["duration"])
                    df = df.assign(**meta)
                    df["exp_id"] = df["platform"]+"\n"+df["experiment"]+"\n"+df["memory"]

                    dfs.append(df)

    df = pd.concat(dfs)

    fig, ax = plt.subplots()
    sb.violinplot(x="exp_id", y="duration", data=df, cut=0)
    # ax.set_xticklabels(platform_names[p] for p in args.platforms)

    ax.set_ylabel("Duration [s]")
    ax.set_xlabel(None)

    plt.tight_layout()
    plt.show()


def line_plot():
    fig, ax = plt.subplots()
    x_axis_len = []

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
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

                    path = os.path.join("/Users/Laurin/Desktop", "res", "runtime", f"{b}-{e}-{m}.pdf")
                    try:
                        plot_func(path)
                    except:
                        pass
                    else:
                        print(path)
    else:
        plot_func()