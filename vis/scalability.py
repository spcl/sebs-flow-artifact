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


if __name__ == "__main__":
    fig, ax = plt.subplots()

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                df = df.groupby("request_id").head(30) # only select the first 30 invos
                assert(len(df["request_id"].unique()) == 30)

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
                    cs.update(cs_end)

                    assert(len(cs_start) + len(cs_end) > 0)

                    xs.append(t)
                    ys.append(len(cs))


                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")

    ax.set_title(f"{args.benchmark} scalability")
    ax.set_xlabel("time [s]")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel("#threads")
    fig.legend()

    plt.show()