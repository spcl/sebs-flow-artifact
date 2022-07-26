import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int)
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-m", "--memory", nargs='+', default=[])
parser.add_argument("-v", "--visualization", choices=["scatter", "line"], default="line")

args = parser.parse_args()

colors = sb.color_palette()
color_map = {
    "AWS": colors[0],
    "AWS Docs": colors[0],
    "Azure": colors[1],
    "Google Cloud": colors[2],
    "Google Cloud Docs": colors[2],
}


platform_names = {
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud"
}

aws_vcpu = {
    128: 1/8,
    1769: 1,
}

gcp_vcpu = {
    128: .083,
    256: .167,
    512: .333,
    1024: .583,
    2048: 1
}

def read(path):
    df = pd.read_csv(path)
    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


def line_plot():
    fig, ax = plt.subplots()
    func = "process"
    data = []

    for idx, platform in enumerate(args.platforms):
        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                path = os.path.join("perf-cost", "640.selfish-detour", platform, filename)
                if not os.path.exists(path):
                    continue

                df = read(path)

                for i in range(df.shape[0]):
                    tps = df.at[i, "result.tps"]
                    tpms = tps/1e3
                    min_diff = (df.at[i, "result.min_diff"])/tpms
                    num_iterations = df.at[i, "result.num_iterations"]
                    res = json.loads(df.at[i, "result.timestamps"])
                    duration = (res[-1] - res[2])/tps
                    func_duration = df.at[i, "end"] - df.at[i, "start"]

                    ys = []
                    for i in range(0, len(res), 2):
                        x = (res[i]-res[0])/tps
                        y = (res[i+1]-res[i]-min_diff)/tpms
                        ys.append(y)

                    ys = np.asarray(ys)
                    k = len(ys)*min_diff
                    p = (np.sum(ys)-k)/duration/1000
                    # print(np.sum(ys), k, duration/1000)
                    assert(p > 0)
                    # print(platform, "total noise:", p, "%")

                    data.append({"name": platform_names[platform],
                                "platform": platform,
                                 "experiment": experiment,
                                 "memory": int(memory),
                                 "suspended": p})

    df = pd.DataFrame(data)
    for mem, vcpu in aws_vcpu.items():
        df = df.append({"platform": "aws", "name": "AWS Docs", "memory": mem, "suspended": 1-vcpu}, ignore_index=True)
    for mem, vcpu in gcp_vcpu.items():
        df = df.append({"platform": "gcp", "name": "Google Cloud Docs", "memory": mem, "suspended": 1-vcpu}, ignore_index=True)
    df = df.sort_values(by="name")

    dashes = [(1, 0), (5, 2), (1, 0), (1, 0), (5, 2)]
    ls = sb.lineplot(data=df, x="memory", y="suspended", hue="name", ci=95, palette=color_map, style="name", dashes=dashes)

    # ax.set_title(f"OS noise ({platform})")
    ax.set_xlabel("Memory configuration [MB]")
    ax.set_ylabel("Suspension time [%]")
    ax.set_xscale("log", base=2)

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_plot():
    func = "process"

    for platform in args.platforms:
        print(platform)
        path = os.path.join("perf-cost", "640.selfish-detour", platform, "sequential_processed.csv")
        if not os.path.exists(path):
            continue

        df = read(path)
        assert(df.shape[0] == 1)

        tps = df.at[0, "result.tps"]
        tpms = tps/1e3
        min_diff = (df.at[0, "result.min_diff"]*900)/tpms
        num_iterations = df.at[0, "result.num_iterations"]
        res = json.loads(df.at[0, "result.timestamps"])
        duration = (res[-1] - res[2])/tps
        func_duration = df.at[0, "end"] - df.at[0, "start"]

        print("min:", min_diff, "number of iterations:", num_iterations, "duration:", f"{duration} s", "func duration:", f"{func_duration}s")
        xs = []
        ys = []
        for i in range(0, len(res), 2):
            x = (res[i]-res[0])/tps
            y = (res[i+1]-res[i]-min_diff)/tpms
            ys.append(y)
            xs.append(x)

        print(min(xs), max(xs))
        print(min(ys), max(ys))
        print(len(xs))

        xs = np.asarray(xs)
        ys = np.asarray(ys)

        k = len(ys)*min_diff
        print("Total noise:", (np.sum(ys)-k)/duration/1000, "%")
        print("======================")

        if args.number:
            idx = np.random.choice(xs.shape[0], args.number)
            xs = xs[idx]
            ys = ys[idx]

        fig, ax = plt.subplots()
        ax.scatter(xs, ys)

        # ax.set_title(f"OS noise ({platform})")
        ax.set_xlabel("Time [s]")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax.set_xlim([0, 5])
        ax.set_ylabel("Time Difference [ms]")
        ax.set_yscale("log")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if args.visualization == "line":
        line_plot()
    else:
        scatter_plot()