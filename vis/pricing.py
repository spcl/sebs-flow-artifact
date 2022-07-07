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
args = parser.parse_args()

num_state_transitions_aws = {
    "650.vid": 5,
    "660.map-reduce": 6,
    "670.auth": 3,
    "680.excamera": 6,
    "690.ml": 4,
}

num_state_transitions_gcp = {
    "650.vid": 5,
    "660.map-reduce": 6,
    "670.auth": 3,
    "680.excamera": 6,
    "690.ml": 4,
}

price_gbs = {
    "aws": 0.0000166667,
}


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

                    df = pd.read_csv(path)
                    df["duration"] = df["end"] - df["start"]
                    per_workflow_price = 0

                    if platform == "aws":
                        assert(np.all(df["billing.duration"] > 0))

                        df["price"] = (df["billing.mem"]/1024 * df["billing.duration"]/1000)*0.0000166667
                        per_workflow_price = (0.000025 * num_state_transitions_aws[benchmark])
                    elif platform == "gcp":
                        num_reqs = df.groupby("request_id").size().mean()

                        df["price"] = (df["billing.mem"]/1024 * np.round(df["duration"], 1))*0.0000025
                        per_workflow_price = (num_reqs * 0.0000004) + (0.00001 * num_state_transitions_gcp[benchmark])
                    else:
                        df["price"] = (df["billing.mem"]/1024 * df["duration"])*0.000016
                        per_workflow_price = (0.2 / 1000000)

                    invos = df.groupby("request_id")
                    df = invos["price"].sum() + per_workflow_price
                    df = df.reset_index(name="price")

                    meta = {"platform": platform, "experiment": experiment, "memory": memory, "benchmark": benchmark}
                    df = df.assign(**meta)
                    df["exp_id"] = df["platform"]+"\n"+df["experiment"]+"\n"+df["memory"]

                    dfs.append(df)

    df = pd.concat(dfs)

    fig, ax = plt.subplots()
    sb.violinplot(x="exp_id", y="price", data=df)
    # ax.set_xticklabels(args.platforms)

    ax.set_ylabel("Price [$]")
    ax.set_xlabel(None)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    violin_plot()