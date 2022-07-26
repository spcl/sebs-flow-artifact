import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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

colors = sb.color_palette()
color_map = {
    "AWS": colors[0],
    "AWS Docs": colors[3],
    "Azure": colors[1],
    "Google Cloud": colors[2],
    "Google Cloud Docs": colors[4],
}

num_state_transitions_aws = {
    "650.vid": 5,
    "660.map-reduce": 6,
    "670.auth": 3,
    "680.excamera": 6,
    "690.ml": 4,
}

num_state_transitions_gcp = {
    "650.vid": 11,
    "660.map-reduce": 24,
    "670.auth": 4,
    "680.excamera": 31,
    "690.ml": 9,
}

num_state_transitions_gcp_profiling = {
    "650.vid": 14,
    "660.map-reduce": 34,
    "670.auth": 4,
    "680.excamera": 43,
    "690.ml": 12,
}

def bar_plot():
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
                    num_reqs = df.groupby("request_id").size().mean()

                    if platform == "aws":
                        assert(np.all(df["billing.duration"] > 0))

                        gb = df["billing.mem"]/1024
                        s = df["billing.duration"]/1000
                        df["price"] = gb*s*0.0000166667

                        # cloud functions + state transitions
                        per_workflow_price = (num_reqs * 0.2/1_000_000)+(0.000025 * num_state_transitions_aws[benchmark])
                    elif platform == "gcp":
                        gb = df["billing.mem"]/1024
                        s = np.round(df["duration"], 1)
                        df["price"] = gb*s*0.0000025

                        # cloud functions + state transitions
                        per_workflow_price = (num_reqs * 0.4/1_000_000)+(0.00001 * num_state_transitions_gcp[benchmark])
                    else:
                        # df["provider.execution_duration"] = df["provider.execution_duration"]/1000.0
                        # df.loc[df["provider.execution_duration"] <= 0, "provider.execution_duration"] = df["duration"]
                        # assert(np.all(df["provider.execution_duration"] > 0))

                        gbs = df["billing.gbs"].max() # it's the same for every row
                        gbs = gbs * df["duration"]

                        df["price"] = gbs*0.000016
                        per_workflow_price = (0.2 / 1000000)

                    print(platform, np.mean(df["price"]/df["duration"]))

                    invos = df.groupby("request_id")
                    df = pd.DataFrame(invos["price"].sum())

                    meta = {"platform": platform, "experiment": experiment, "memory": memory, "benchmark": benchmark, "price_per_workflow": per_workflow_price}
                    df = df.assign(**meta)
                    df["exp_id"] = df["platform"]+"\n"+df["experiment"]+"\n"+df["memory"]

                    dfs.append(df)

    df = pd.concat(dfs)
    fig, ax = plt.subplots()

    w = 0.3
    xs = np.arange(len(args.experiments))

    for idx, platform in enumerate(args.platforms):
        ys = []
        es = []
        o = ((len(args.platforms)-1)*w)/2.0 - idx*w

        for e in args.experiments:
            vals = df.loc[(df["platform"] == platform) & (df["experiment"] == e), "price"]
            y = vals.mean()

            print(platform, e, y)

            e = st.t.interval(alpha=0.95, df=len(vals)-1, loc=y, scale=st.sem(vals))
            e = np.maximum(e, 0)
            e = np.abs(e-y)

            ys.append(y)
            es.append(e)

        es = np.asarray(es)
        es = np.reshape(es, [2, -1])

        name = platform_names[platform]
        ax.bar(xs-o, ys, w, color=color_map[name], yerr=es, capsize=3, label=name)

        price_per_workflow = df.loc[df["platform"] == platform, "price_per_workflow"].max()
        ax.bar(xs-o, len(ys)*[price_per_workflow], w, color=color_map[name], alpha=0.7, bottom=ys)

    # ax.bar(xs, np.zeros_like(xs), width=0, color="0", label="GBs")
    # ax.bar(xs, np.zeros_like(xs), width=0, color="0", alpha=0.7, label="Transitions")

    ax.set_xticks(xs, args.experiments)

    ax.set_ylabel("Price [$]")
    ax.set_xlabel(None)
    fig.legend(bbox_to_anchor=(0.97, 0.97))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bar_plot()