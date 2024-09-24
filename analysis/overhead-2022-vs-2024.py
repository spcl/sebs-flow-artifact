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

noise_256 = {"aws": 0.854, "azure": 0.49, "gcp": 0.754}
noise_1024 = {"aws": 0.4282, "azure": 0.49, "gcp": 0.1558}
noise = {
    256: noise_256,
    1024: noise_1024
}

platform_names = {
    "aws/2022": "2022",
    "azure/2022": "2022",
    "gcp/2022": "2022",
    "aws/2024": "2024",
    "gcp/2024": "2024",
    "azure/2024": "2024",
    "azure_2e10": "2022",
    "azure_2e10/2024": "2024",
    "azure_2e28": "2022",
    "azure_2e28/2024": "2024"
}


colors = sb.color_palette("colorblind")
#colors for 2022 vs 2024 comparison. 
color_map = {
    "2022": colors[0],
    "2024": colors[1]
}


def read(path):
    df = pd.read_csv(path)
    req_ids = df["request_id"].unique()
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
        
        ds_noise = []
        cs_noise = []
        dse_noise = []
        cse_noise = []

        for experiment in args.experiments:
            for memory in args.memory:
                filename = f"{experiment}_{memory}_processed.csv" if platform != "azure/2022" and platform != "azure_2e10" and platform != "azure_2e28" else f"{experiment}.csv"
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
                    print(df)
                invos = df.groupby("request_id")

                
                d_total = invos["end"].max() - invos["start"].min()

                #fix for 1000genomes: due to parallel stage, d_total and d_critical have to be computed differently. 
                if args.benchmark == "6100.1000-genome":
                    #compute critical path differently due to parallel stage. 
                    invos = df.groupby(["request_id", "phase"])
                    d_critical = invos["duration"].max().groupby("request_id").sum()
                else:    
                    invos = df.groupby(["request_id", "func"])
                    d_critical = invos["duration"].max().groupby("request_id").sum()

                if args.noise:
                    d_noise = noise[int(memory)][platform.split("/")[0]] * d_critical
                    d_critical_noise = d_critical - d_noise
                    d_total_noise = d_total + d_noise
                    print("noise")

                assert(np.all(d_critical <= d_total))

                print(platform, experiment, memory)
                print("total avg:", np.mean(d_total))
                print("execution avg:", np.mean(d_critical))
                print("overhead avg:", np.mean(d_total - d_critical))
                print("cold starts:", df["is_cold"].mean())
                
                if args.noise:
                    print("critical normalized:", np.mean(d_critical_noise))

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
                
                if args.noise:
                    d_noise = np.mean(d_total_noise)
                    de_noise = st.t.interval(confidence=0.95, df=len(d_total_noise)-1, loc=d_noise, scale=st.sem(d_total_noise))
                    de_noise = np.abs(de_noise-d_noise)

                    ds_noise.append(d_noise)
                    dse_noise.append(de_noise)

                c = np.mean(d_critical)
                ce = st.t.interval(confidence=0.95, df=len(d_critical)-1, loc=c, scale=st.sem(d_critical))
                ce = np.abs(ce-c)
                cs.append(c)
                cse.append(ce)
                
                if args.noise:
                    c_noise = np.mean(d_critical_noise)
                    ce_noise = st.t.interval(confidence=0.95, df=len(d_critical_noise)-1, loc=c_noise, scale=st.sem(d_critical_noise))
                    ce_noise = np.abs(ce_noise-c_noise)
                    cs_noise.append(c_noise)
                    cse_noise.append(ce_noise)

        ds = np.asarray(ds)
        cs = np.asarray(cs)
        dse = np.asarray(dse).transpose()
        cse = np.asarray(cse).transpose()
        
        ds_noise = np.asarray(ds_noise)
        cs_noise = np.asarray(cs_noise)
        dse_noise = np.asarray(dse_noise).transpose()
        cse_noise = np.asarray(cse_noise).transpose()

        if "2022" in platform:
            o = .1
        elif "2024" in platform:
            o = -.1
        else:
            o = .1
            
        if "gcp" in platform:
            at = .5
        elif "aws" in platform:
            at = 1
        else:
            at = 1.5
        
        name = platform_names[platform]

        color = color_map[name]

        bar = ax.bar((at)-o, cs, w, yerr=cse, label=name, color=color, capsize=3, linewidth=1)
        
        if args.noise:
            noise_name = name + "normalized"
            color = color_map["2022"]
            bar = ax.bar((at)-.1, cs_noise, w, yerr=cse_noise, label=noise_name, color=color, capsize=3, linewidth=1)
        
        if not args.no_overhead:
            ax.bar((at)-o, ds-cs, w, yerr=dse, capsize=3, bottom=cs, alpha=0.7, color=color, hatch='///')
    ax.set_ylabel("Duration [s]",fontsize=16)
    
    ax.set_xticks(xs, ['', 'Google Cloud', 'AWS', 'Azure'], fontsize=16)

      
    if args.noise:
        colors = {'Normalized':colors[0], 'Original':colors[1]}  
        labels = ['Normalized', 'Original']
    else:
        colors = {'2022':colors[0], '2024':colors[1]}  
        labels = ['2022', '2024']
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, fontsize=16)
    
    #fig.legend(bbox_to_anchor=(0.97, 0.97))
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if args.noise:
        plt.savefig("../figures/plots/overhead/overhead-osnoise-" + args.benchmark + ".pdf")
    plt.savefig("../figures/plots/2022-vs-2024/2022-vs-2024-" + args.benchmark + ".pdf")
    plt.show()

if __name__ == "__main__":
    bar_plot()
