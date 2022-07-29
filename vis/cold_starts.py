import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-e", "--experiment", default="burst")
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

def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:30]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    if np.any(df["duration"] <= 0):
        raise ValueError(f"Invalid data frame at path {path}")

    return df


def bar_plot():
    fig, ax = plt.subplots()
    w = 0.3

    benchmarks = ["650.vid", "660.map-reduce", "670.auth", "680.excamera", "690.ml"]
    memory = [256, 1024]

    xs = np.arange(len(benchmarks))

    for idx, platform in enumerate(args.platforms):
        ys = []
        es = []

        for benchmark in benchmarks:
            for m in memory:
                filename = f"{args.experiment}_{m}_processed.csv" if platform != "azure" else f"{args.experiment}_processed.csv"
                path = os.path.join("perf-cost", benchmark, platform, filename)
                if not os.path.exists(path):
                    print(path)
                    continue

                df = read(path)

                cold = 100*df["is_cold"]
                y = cold.mean()
                e = st.t.interval(alpha=0.95, df=len(cold)-1, loc=y, scale=st.sem(cold))
                e = np.minimum(e, 100)
                e = np.maximum(e, 0)
                e = np.abs(e-y)

                ys.append(y)
                es.append(e)
                break

        es = np.asarray(es).transpose()

        name = platform_names[platform]
        o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        ax.bar(xs-o, ys, w, label=name, capsize=3, color=color_map[name], yerr=es)

    ax.set_ylabel("Cold starts [%]")
    ax.set_xticks(xs, ["Video Analysis", "MapReduce", "Oracle", "ExCamera", "Machine Learning"])
    fig.legend(bbox_to_anchor=(0.97, 0.97))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bar_plot()