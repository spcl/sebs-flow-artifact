import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", type=str)
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-m", "--memory", nargs='+', default=[])
parser.add_argument("-v", "--visualization", choices=["bar", "violin", "line"], default="bar")
parser.add_argument("--sum", action="store_true", default=False)
args = parser.parse_args()


def read(path):
    df = pd.read_csv(path)
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

                ys.append(np.mean(d_total))
                es.append(np.std(d_total))

        o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        ax.bar(xs-o, ys, w, label=platform, yerr=es, capsize=3)

    ax.set_title(f"{args.benchmark} Runtime")
    ax.set_ylabel("Duration [s]")
    ax.set_xticks(xs, args.experiments)
    fig.legend()

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

                    df = pd.DataFrame(data, columns=["duration"])
                    df = df.assign(**meta)
                    df["exp_id"] = df["platform"]+"\n"+df["experiment"]+"\n"+df["memory"]

                    dfs.append(df)

    df = pd.concat(dfs)

    if len(benchmarks) == 1:
        fig, ax = plt.subplots()
        sb.violinplot(x="exp_id", y="duration", data=df)
        # ax.set_xticklabels(args.platforms)
    else:
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 3)

        ax = fig.add_subplot(gs[0, 0])
        sb.violinplot(x="exp_id", y="duration", data=df.loc[df["benchmark"] == "650.vid"])

        ax = fig.add_subplot(gs[0, 1])
        sb.violinplot(x="exp_id", y="duration", data=df.loc[df["benchmark"] == "660.map-reduce"])

        ax = fig.add_subplot(gs[0, 2])
        sb.violinplot(x="exp_id", y="duration", data=df.loc[df["benchmark"] == "670.auth"])

        ax = fig.add_subplot(gs[1, 0])
        sb.violinplot(x="exp_id", y="duration", data=df.loc[df["benchmark"] == "680.excamera"])

        ax = fig.add_subplot(gs[1, 1])
        sb.violinplot(x="exp_id", y="duration", data=df.loc[df["benchmark"] == "690.ml"])

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
    if args.visualization == "bar":
        bar_plot()
    elif args.visualization == "violin":
        violin_plot()
    else:
        line_plot()