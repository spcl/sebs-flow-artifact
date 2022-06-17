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
parser.add_argument("--mean", action="store_true", default=False)
args = parser.parse_args()


def bar_plot():
    fig, ax = plt.subplots()
    w = 0.3
    xs = np.arange(len(args.experiments))

    for idx, platform in enumerate(args.platforms):
        ys = []
        es = []

        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    ys.append(0)
                    es.append(0)
                    continue

                df = pd.read_csv(path)
                invos = df.groupby("request_id")
                d_total = invos["end"].max() - invos["start"].min()

                ys.append(np.mean(d_total))
                es.append(np.std(d_total))

        o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        ax.bar(xs-o, ys, w, label=platform, yerr=es, capsize=3)

    ax.set_title(f"{args.benchmark} runtime")
    ax.set_ylabel("duration [s]")
    ax.set_xticks(xs, args.experiments)
    fig.legend()

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

                df = pd.read_csv(path)
                invos = df.groupby("request_id")
                d_total = invos["end"].max() - invos["start"].min()

                ys = np.asarray(d_total)
                xs = np.arange(ys.shape[0])

                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")

                ys = ys[np.where(~np.isnan(ys))]
                print(platform, "std:", np.std(ys))
                x_axis_len.append(len(xs))

    ax.set_title(f"{args.benchmark} runtime")
    ax.set_xlabel("repetition")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, min(x_axis_len)+1, 5))
    ax.set_ylabel("duration [s]")
    ax.set_xlim([0, min(x_axis_len)-1])
    fig.legend()

    plt.show()


if __name__ == "__main__":
    if args.mean:
        bar_plot()
    else:
        line_plot()