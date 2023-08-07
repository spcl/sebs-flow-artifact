import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


size_mapping = {
    "2e5": 2**5,
    "2e8": 2**8,
    "2e10": 2**10,
    "2e12": 2**12,
    "2e14": 2**14,
    "2e16": 2**16,
    "2e20": 2**20,
    "2e25": 2**25,
    "2e26": 2**26,
    "2e27": 2**27,
    "2e28": 2**28,
}

platform_names = {
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud"
}


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-s", "--sizes", nargs='+', default=list(size_mapping.keys()))
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
args = parser.parse_args()


def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:30]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


def line_plot():
    func = "process"
    fig, ax = plt.subplots()
    dfs = []

    for platform in args.platforms:
        for size in args.sizes:
            for experiment in args.experiments:
                filename = f"{experiment}_512.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", "631.parallel-download", f"{platform}_{size}", filename)
                if not os.path.exists(path):
                    continue

                df = read(path)

                invos = df.groupby("request_id")
                d_total = invos["end"].max() - invos["start"].min()

                invos = df.groupby(["request_id", "func"])
                d_critical = invos["duration"].max().groupby("request_id").sum()

                print(platform, size, np.mean(d_total), np.mean(d_total - d_critical))

                data = {"request_id": d_critical.index,
                        "overhead": d_total - d_critical,
                        "experiment": experiment,
                        "platform": platform_names[platform],
                        "io": size_mapping[size]}
                dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, ignore_index=True)
    sb.lineplot(data=df, x="io", y="overhead", hue="platform", ci=95)

    # ax.set_title("function invocation")
    ax.set_ylabel("Overhead [s]")
    ax.set_xlabel("Download [b]")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    line_plot()