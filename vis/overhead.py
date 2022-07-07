import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", type=str)
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-m", "--memory", nargs='+', default=[])
parser.add_argument("--no-overhead", action="store_true", default=False)
parser.add_argument("--noise", action="store_true", default=False)
parser.add_argument("-v", "--visualization", choices=["bar", "line"], default="bar")
args = parser.parse_args()

noise = {"aws": 0.927, "azure": 0.414, "gcp": 0.879}


def read(path):
    df = pd.read_csv(path)
    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


def bar_plot():
    fig, ax = plt.subplots()
    w = 0.3
    xs = np.arange(len(args.experiments))

    for idx, platform in enumerate(args.platforms):
        ds = []
        cs = []
        dse = []
        cse = []

        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    ds.append(0)
                    dse.append(0)
                    cs.append(0)
                    cse.append(0)
                    continue

                df = read(path)
                invos = df.groupby("request_id")

                d_total = invos["end"].max() - invos["start"].min()

                invos = df.groupby(["request_id", "func"])
                d_critical = invos["duration"].max().groupby("request_id").sum()

                if args.noise:
                    d_noise = noise[platform] * d_critical
                    d_critical -= d_noise
                    d_total += d_noise

                assert(np.all(d_critical < d_total))

                print(platform, experiment, memory, "total avg:", np.mean(d_total))
                print(platform, experiment, memory, "execution avg:", np.mean(d_critical))
                print(platform, experiment, memory, "overhead avg:", np.mean(d_total - d_critical))

                ds.append(np.mean(d_total))
                dse.append(np.std(d_total))
                cs.append(np.mean(d_critical))
                cse.append(np.std(d_critical))

        ds = np.asarray(ds)
        cs = np.asarray(cs)

        o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        bar = ax.bar(xs-o, cs, w, yerr=cse, label=platform, capsize=3, linewidth=1)
        if not args.no_overhead:
            color = bar.patches[0].get_facecolor()
            ax.bar(xs-o, ds-cs, w, yerr=dse, capsize=3, bottom=cs, alpha=0.7, color=color)

    # ax.set_title(f"{args.benchmark} overhead")
    ax.set_ylabel("duration [s]")
    ax.set_xticks(xs, args.experiments)
    fig.legend()

    plt.tight_layout()
    plt.show()


def line_plot():
    fig, ax = plt.subplots()
    x_axis_len = []

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    continue

                df = read(path)
                invos = df.groupby("request_id")

                d_total = invos["end"].max() - invos["start"].min()

                invos = df.groupby(["request_id", "func"])
                d_critical = invos["duration"].max().groupby("request_id").sum()

                assert(np.all(d_critical <= d_total))
                d_overhead = d_total - d_critical

                ys = np.asarray(d_overhead)
                xs = np.arange(ys.shape[0])

                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")

                x_axis_len.append(len(xs))

    ax.set_title(f"{args.benchmark} overhead")
    ax.set_xlabel("repetition")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, min(x_axis_len)+1, 5))
    ax.set_ylabel("overhead [s]")
    ax.set_xlim([0, min(x_axis_len)-1])
    fig.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if args.visualization == "bar":
        bar_plot()
    else:
        line_plot()