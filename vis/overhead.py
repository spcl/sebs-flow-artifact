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
parser.add_argument("--no-overhead", action="store_true", default=False)
parser.add_argument("--noise", action="store_true", default=False)
parser.add_argument("-v", "--visualization", choices=["bar", "line", "violin"], default="bar")
args = parser.parse_args()

noise_256 = {"aws": 0.853, "azure": 0.27, "gcp": 0.693}
noise_1024 = {"aws": 0.4000, "azure": 0.27, "gcp": 0.1505}
noise = {
    256: noise_256,
    1024: noise_1024
}

platform_names = {
    "aws/laurin": "AWS 2022",
    "azure/laurin": "Azure 2022",
    "gcp/laurin": "Google Cloud 2022",
    "aws/batch-size-30-reps-6": "AWS",
    "gcp/batch-size-30-reps-6": "Google Cloud",
    "azure/batch-size-30-reps-6": "Azure"
}

platform_names_old = {
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud"
}

colors = sb.color_palette("colorblind")
color_map = {
    "AWS": colors[0],
    "AWS Docs": colors[3],
    "Azure": colors[1],
    "Google Cloud": colors[2],
    "Google Cloud Docs": colors[4],
}

def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()[:180]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    return df


def bar_plot():
    sb.set_theme()
    sb.set_context("paper")
    colors = sb.color_palette("colorblind")
    fig, ax = plt.subplots()
    w = 0.2
    xs = np.arange(2,step=.5)

    for idx, platform in enumerate(args.platforms):
        ds = []
        cs = []
        dse = []
        cse = []

        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                path = os.path.join("./../perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    print(path)
                    ds.append(0)
                    dse.append([0, 0])
                    cs.append(0)
                    cse.append([0, 0])
                    continue

                df = read(path)
                if args.benchmark == "6100.1000-genome":
                    df.loc[df.func == "individuals", "phase"] = "phase0"
                    df.loc[df.func == "individuals_merge", "phase"] = "phase1"
                    df.loc[df.func == "sifting", "phase"] = "phase1"
                    df.loc[df.func == "frequency", "phase"] = "phase2"
                    df.loc[df.func == "mutation_overlap", "phase"] = "phase2"
                    #print(df)                
                invos = df.groupby("request_id")

                d_total = invos["end"].max() - invos["start"].min()

                #invos = df.groupby(["request_id", "func"])
                #d_critical = invos["duration"].max().groupby("request_id").sum()

                if args.benchmark == "6100.1000-genome":
                    #compute critical path differently due to parallel stage. 
                    invos = df.groupby(["request_id", "phase"])
                    d_critical = invos["duration"].max().groupby("request_id").sum()
                else:    
                    invos = df.groupby(["request_id", "func"])
                    d_critical = invos["duration"].max().groupby("request_id").sum()

                if args.noise:
                    d_noise = noise[int(memory)][platform.split("/")[0]] * d_critical
                    d_critical -= d_noise
                    d_total += d_noise

                assert(np.all(d_critical < d_total))

                print(platform, experiment, memory)
                print("total avg:", np.mean(d_total))
                print("execution avg:", np.mean(d_critical))
                print("overhead avg:", np.mean(d_total - d_critical))
                print("cold starts:", df["is_cold"].mean())

                df = df.sort_values("start")
                
                if args.benchmark == "6100.1000-genome":
                    funcs = df["phase"].unique()
                else:
                    funcs = df["func"].unique()
                    
                for a, b in zip(funcs[:-1], funcs[1:]):
                    if args.benchmark == "6100.1000-genome":
                        tmp = invos.agg({"end": np.amax, "start": np.amin}).reset_index(["request_id", "phase"]).set_index("request_id")
                        end_a = tmp.loc[tmp["phase"] == a, "end"]
                        start_b = tmp.loc[tmp["phase"] == b, "start"]
                    else:
                        tmp = invos.agg({"end": np.amax, "start": np.amin}).reset_index(["request_id", "func"]).set_index("request_id")
                        end_a = tmp.loc[tmp["func"] == a, "end"]
                        start_b = tmp.loc[tmp["func"] == b, "start"]

                    print("latency", a, "->", b, ":", np.mean(start_b - end_a))

                print("======================================")

                d = np.mean(d_total)
                de = st.t.interval(confidence=0.95, df=len(d_total)-1, loc=d, scale=st.sem(d_total))
                de = np.abs(de-d)

                ds.append(d)
                dse.append(de)

                c = np.mean(d_critical)
                ce = st.t.interval(confidence=0.95, df=len(d_critical)-1, loc=c, scale=st.sem(d_critical))
                ce = np.abs(ce-c)
                cs.append(c)
                cse.append(ce)

        ds = np.asarray(ds)
        cs = np.asarray(cs)
        dse = np.asarray(dse).transpose()
        cse = np.asarray(cse).transpose()

        o = 0
        if "gcp" in platform:
            at = .5
        elif "aws" in platform:
            at = 1
        else:
            at = 1.5
        #o = ((len(args.platforms)-1)*w)/2.0 - idx*w
        name = platform_names[platform]
        color = color_map[name]
        bar = ax.bar(at, cs, w, yerr=cse, label=name, color=color, capsize=3, linewidth=1)
        if not args.no_overhead:
            ax.bar(at, ds-cs, w, yerr=dse, capsize=3, bottom=cs, alpha=0.7, color=color, hatch='///')

    # ax.set_title(f"{args.benchmark} overhead")
    ax.set_ylabel("Duration [s]",fontsize=16)
    #ax.set_xticks(xs, args.experiments, fontsize=16)
    xlabels = ['', 'Google Cloud', 'AWS', 'Azure']
    ax.set_xticks(xs, xlabels, fontsize=16)
    fig.legend(bbox_to_anchor=(0.97, 0.97),fontsize=16)
    #fig.legend(bbox_to_anchor=(0.55, 0.97),fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    #if args.noise:
        #plt.savefig("../figures/plots/overhead/overhead-osnoise-" + args.benchmark + ".pdf")
    plt.savefig("../figures/plots/overhead/overhead-" + args.benchmark + ".pdf")
    plt.show()


def line_plot():
    fig, ax = plt.subplots()
    x_axis_len = []

    for platform in args.platforms:
        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure" else f"{experiment}_processed.csv"
                path = os.path.join("perf-cost", args.benchmark, platform, filename)
                if not os.path.exists(path):
                    continue

                df = read(path)
                invos = df.groupby("request_id")

                d_total = invos["end"].max() - invos["start"].min()

                invos = df.groupby(["request_id", "func"])
                d_critical = invos["duration"].max().groupby("request_id").sum()

                assert(np.all(d_critical <= d_total))
                d_overhead = d_total - d_critical

                ys = np.asarray(d_overhead)
                xs = np.arange(ys.shape[0])

                line = ax.plot(xs, ys)[0]
                line.set_label(f"{platform}_{experiment}_{memory}")

                x_axis_len.append(len(xs))

    ax.set_title(f"{args.benchmark} overhead")
    ax.set_xlabel("repetition")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(np.arange(0, min(x_axis_len)+1, 5))
    ax.set_ylabel("overhead [s]")
    ax.set_xlim([0, min(x_axis_len)-1])
    fig.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if args.visualization == "bar":
        bar_plot()
    elif args.visualization == "violin":
        violin_plot()
    else:
        line_plot()
