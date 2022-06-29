import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
    parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
    parser.add_argument("-m", "--memory", nargs='+', default=[])
    args = parser.parse_args()

    func = "process"
    fig, ax = plt.subplots()
    x_axis_len = []

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                path = os.path.join("perf-cost", "620.func-invo", platform, f"{experiment}_{memory}.csv")
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                req_ids = df["request_id"].unique()

                ys = []
                for id in req_ids:
                    start = df.loc[((df["func"] == func) & (df["request_id"] == id))].sort_values(["start"])["start"].to_numpy()
                    end = df.loc[((df["func"] == func) & (df["request_id"] == id))].sort_values(["end"])["end"].to_numpy()

                    # sanity checks to verify no functions are overlapping
                    assert(np.all(start[:-1] < start[1:]))
                    assert(np.all(end[:-1] < end[1:]))
                    assert(np.all(end[:-1] < start[1:]))

                    ys.append(start[1:] - end[:-1])

                ys = np.asarray(ys)
                ys = np.mean(ys, axis=1)
                xs = np.arange(ys.shape[0])

                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")
                x_axis_len.append(len(xs))

    ax.set_title("function invocation")
    ax.set_xlabel("repetition")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, len(xs)+1, 5))
    ax.set_ylabel("latency [s]")
    ax.set_xlim([0, min(x_axis_len)-1])
    fig.legend()

    plt.show()