import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiments", nargs='+', default=["burst", "cold", "sequential", "warm"])
parser.add_argument("-s", "--sizes", nargs='+', default=["2e5", "2e8", "2e10", "2e12", "2e14", "2e16", "2e18-1000"])
parser.add_argument("-p", "--platforms", nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-v", "--visualization", choices=["violin", "line"], default="violin")
args = parser.parse_args()

size_mapping = {
    "2e5": 2**5,
    "2e8": 2**8,
    "2e10": 2**10,
    "2e12": 2**12,
    "2e14": 2**14,
    "2e16": 2**16,
    "2e18-1000": 2**18-1000
}

platform_names = {
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud"
}

def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:30]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]

    req_ids = df["request_id"].unique()
    req_ids = req_ids[:30]
    df = df.loc[df["request_id"].isin(req_ids)]

    return df


def line_plot():
    func = "process"
    fig, ax = plt.subplots()
    dfs = []

    for platform in args.platforms:
        for size in args.sizes:
            for experiment in args.experiments:
                filename = f"{experiment}_256.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", "620.func-invo", f"{platform}_{size}", filename)
                if not os.path.exists(path):
                    print(path)
                    continue


                df = read(path)
                req_ids = df["request_id"].unique()
                assert(len(req_ids) == 30)

                ls = []
                rs = []
                for id in req_ids:
                    start = df.loc[((df["func"] == func) & (df["request_id"] == id))].sort_values(["start"])["start"].to_numpy()
                    end = df.loc[((df["func"] == func) & (df["request_id"] == id))].sort_values(["end"])["end"].to_numpy()

                    # sanity checks to verify no functions are overlapping
                    assert(np.all(start[:-1] < start[1:]))
                    assert(np.all(end[:-1] < end[1:]))
                    assert(np.all(end[:-1] < start[1:]))

                    ds = start[1:] - end[:-1]
                    ls += ds.tolist()
                    rs += len(ds) * [id]

                data = {"request_id": rs,
                        "latency": ls,
                        "experiment": experiment,
                        "platform": platform_names[platform],
                        "payload_size": size_mapping[size]}
                dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, ignore_index=True)
    sb.lineplot(data=df, x="payload_size", y="latency", hue="platform", ci=95)

    # ax.set_title("function invocation")
    ax.set_ylabel("Latency [s]")
    ax.set_xlabel("Payload size [b]")
    ax.set_xscale("log", base=2)
    # ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.tight_layout()
    plt.show()


def violin_plot():
    func = "process"
    fig, ax = plt.subplots()
    dfs = []

    for platform in args.platforms:
        for experiment in args.experiments:
            filename = f"{experiment}_128_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
            path = os.path.join("perf-cost", "620.func-invo", platform, filename)
            if not os.path.exists(path):
                continue

            df = read(path)
            req_ids = df["request_id"].unique()

            ls = []
            rs = []
            for id in req_ids:
                start = df.loc[((df["func"] == func) & (df["request_id"] == id))].sort_values(["start"])["start"].to_numpy()
                end = df.loc[((df["func"] == func) & (df["request_id"] == id))].sort_values(["end"])["end"].to_numpy()

                # sanity checks to verify no functions are overlapping
                assert(np.all(start[:-1] < start[1:]))
                assert(np.all(end[:-1] < end[1:]))
                assert(np.all(end[:-1] < start[1:]))

                ds = start[1:] - end[:-1]
                ls += ds.tolist()
                rs += len(ds) * [id]

            data = {"request_id": rs, "latency": ls, "experiment": experiment, "platform": platform}
            df = pd.DataFrame(data)
            df["exp_id"] = df["platform"]+"\n"+df["experiment"]
            dfs.append(df)

    df = pd.concat(dfs)
    sb.violinplot(x="exp_id", y="latency", data=df)
    ax.set_xlabel(None)

    fig.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if args.visualization == "line":
        line_plot()
    else:
        violin_plot()