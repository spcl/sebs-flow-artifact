import os
import argparse
import json
import dateutil.parser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--benchmark", type=str)
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-m", "--memory", nargs='+', default=[])
args = parser.parse_args()


def read(path):
    df = pd.read_csv(path)
    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


if __name__ == "__main__":
    fig, ax = plt.subplots()

    azure_experiments = set()

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                if platform == "azure":
                    if experiment in azure_experiments:
                        continue
                    else:
                        azure_experiments.add(experiment)

                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    continue

                df = read(path)
                reqs = list(df.groupby("request_id"))[:30]
                df = pd.concat(g[1] for g in reqs) # only select the first 30 invos

                req_ids = df["request_id"].unique()
                assert(len(req_ids) == 30)

                # check if they have all been invoked simultaneously
                if experiment == "burst":
                    filename = f"{experiment}_results_{memory}.json" if platform != "azure" else f"{experiment}_results.json"
                    path = os.path.join("perf-cost", args.benchmark, platform, filename)
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
                    assert(diff < 1)

                exp_start = df["start"].min()

                df["start"] -= exp_start
                df["end"] -= exp_start

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

                print(platform, "duration:", df["end"].max(), "max threads:", max(ys))
                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")

    ax.set_title(f"{args.benchmark} scalability")
    ax.set_xlabel("time [s]")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel("#threads")
    ax.set_xscale("log")
    fig.legend()

    plt.tight_layout()
    plt.show()