import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


threads = [2, 4, 8, 16]
durations = [1, 5, 10, 15, 20]

platform_names = {
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud"
}


def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:30]
    assert(len(req_ids) == 30)
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


def matrix_plot():
    func = "process"
    fig, ax = plt.subplots()
    data = np.zeros((len(threads), len(durations), 2))

    for idx_t, t in enumerate(threads):
        for idx_d, d in enumerate(durations):
            path = os.path.join("./../perf-cost", "630.parallel-sleep", f"gcp_{t}t_{d}s", "burst_256_processed.csv")
            if not os.path.exists(path):
                continue

            df = read(path)

            invos = df.groupby("request_id")
            d_total = invos["end"].max() - invos["start"].min()

            invos = df.groupby(["request_id", "func"])
            d_critical = invos["duration"].max().groupby("request_id").sum()

            d_overhead = d_total-d_critical

            num_threads = invos["container_id"].nunique().agg({"container_id": np.mean})

            print(t, d, np.mean(d_total), np.mean(d_critical), np.mean(d_total/d_critical), num_threads-1)
            data[len(threads)-1-idx_t, idx_d, 0] = np.mean(d_total/d_critical)
            data[len(threads)-1-idx_t, idx_d, 1] = np.floor(num_threads-1)

    sb.set(font_scale=1.4)
    sb.heatmap(data[..., 0], xticklabels=durations, yticklabels=threads[::-1], cmap=sb.cm.rocket_r, annot=True)

    ax.set_ylabel("#Functions",fontsize=18)
    ax.set_xlabel("Duration [s]",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()

    plt.savefig("./../figures/plots/overhead/overhead-parallel-burst-gcp.pdf")

    plt.show()


if __name__ == "__main__":
    matrix_plot()
