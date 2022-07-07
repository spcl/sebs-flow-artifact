import os
import argparse
import json
import dateutil.parser

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
parser.add_argument("-v", "--visualization", choices=["bar", "violin", "line"], default="bar")
args = parser.parse_args()


def read(path):
    df = pd.read_csv(path)
    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


def bar_plot():
    fig, ax = plt.subplots()
    w = 0.3
    xs = np.arange(len(args.experiments))

    for idx, platform in enumerate(args.platforms):
        ys = []
        es = []

        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    ys.append(0)
                    es.append(0)
                    continue

                df = read(path)

                filename = f"{experiment}_results_{memory}.json" if platform != "azure" else f"{experiment}_results.json"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                with open(path) as f:
                    results = json.load(f)

                ts = []
                invos = results["_invocations"]
                invos = invos[list(invos.keys())[0]]
                for req_id, data in invos.items():
                    client_begin = data["times"]["client_begin"]
                    client_begin = dateutil.parser.parse(client_begin).timestamp()

                    begin = df.loc[(df["request_id"] == req_id)]["start"].min()
                    t = begin - client_begin
                    if platform != "azure":
                        t -= 2*3600
                    ts.append(t)

                ys.append(np.mean(ts))
                es.append(np.std(ts))

        o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        ax.bar(xs-o, ys, w, label=platform, yerr=es, capsize=3)

    ax.set_title(f"{args.benchmark} scheduling")
    ax.set_ylabel("latency [s]")
    ax.set_xticks(xs, args.experiments)
    fig.legend()

    plt.tight_layout()
    plt.show()


def line_plot():
    fig, ax = plt.subplots()
    x_axis_len = []

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    continue

                df = read(path)
                invos = df.groupby("request_id")
                d_total = invos["end"].max() - invos["start"].min()

                ys = np.asarray(d_total)
                xs = np.arange(ys.shape[0])

                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")

                ys = ys[np.where(~np.isnan(ys))]
                print(platform, "std:", np.std(ys))
                x_axis_len.append(len(xs))

    ax.set_title(f"{args.benchmark} scheduling")
    ax.set_ylabel("latency [s]")
    ax.set_xlabel("repetition")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, min(x_axis_len)+1, 5))
    ax.set_xlim([0, min(x_axis_len)-1])
    fig.legend()

    plt.tight_layout()
    plt.show()


def violin_plot():
    benchmarks = [args.benchmark]
    if not benchmarks[0]:
        benchmarks = [p for p in os.listdir("perf-cost") if os.path.isdir(os.path.join("perf-cost", p))]
        benchmarks.remove("640.selfish-detour")
        benchmarks.remove("620.func-invo")
        benchmarks.remove("630.parallel-sleep")

    dfs = []
    for benchmark in benchmarks:
        for platform in args.platforms:
            memory_confs = args.memory
            if platform == "azure":
                memory_confs = [None]
            elif not memory_confs:
                files = os.listdir(os.path.join("perf-cost", benchmark, platform))
                memory_confs = {mem for name in files for mem in name.split("_") if mem.isnumeric()}

            for experiment in args.experiments:
                for memory in memory_confs:
                    filename = f"{experiment}_{memory}.csv" if platform != "azure" else f"{experiment}.csv"
                    path = os.path.join("perf-cost", benchmark, platform, filename)
                    if not os.path.exists(path):
                        continue

                    df = read(path)

                    filename = f"{experiment}_results_{memory}.json" if platform != "azure" else f"{experiment}_results.json"
                    path = os.path.join("perf-cost", benchmark, platform, filename)
                    with open(path) as f:
                        results = json.load(f)

                    ls = []
                    req_ids = []
                    invos = results["_invocations"]
                    invos = invos[list(invos.keys())[0]]
                    for req_id, data in invos.items():
                        client_begin_raw = data["times"]["client_begin"]
                        client_begin = dateutil.parser.parse(client_begin_raw).timestamp()

                        begin = df.loc[(df["request_id"] == req_id)]["start"].min()
                        t = begin - client_begin - 2*3600
                        if t < 0:
                            t = begin - client_begin

                        if t > 1000 or t < 0:
                            print(f"{filename} {platform} invalid time: {t}")
                            continue
                        ls.append(t)
                        req_ids.append(req_id)

                    data = {"request_id": req_ids, "latency": ls, "experiment": experiment, "platform": platform}
                    if platform == "azure":
                        data["memory"] = "n/a"
                    else:
                        data["memory"] = memory

                    df = pd.DataFrame(data)
                    df["exp_id"] = df["platform"]+"\n"+df["experiment"]+"\n"
                    dfs.append(df)

    df = pd.concat(dfs)

    fig, ax = plt.subplots()
    sb.violinplot(x="exp_id", y="latency", data=df)
    ax.set_ylabel("Duration [s]")
    ax.set_xlabel(None)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if args.visualization == "bar":
        bar_plot()
    elif args.visualization == "violin":
        violin_plot()
    else:
        line_plot()