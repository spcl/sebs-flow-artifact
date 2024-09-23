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
    #set to 30 for 1000genomes on Azure due to fewer repetitions there. 
    req_ids = df["request_id"].unique()[:180]
    df = df.loc[df["request_id"].isin(req_ids)]

    df = df[df["func"] != "run_workflow"]
    df["duration"] = df["end"] - df["start"]
    if np.any(df["duration"] <= 0):
        raise ValueError(f"Invalid data frame at path {path}")

    return df


def bar_plot():
    sb.set_theme()
    sb.set_context("paper")
    fig, ax = plt.subplots()
    w = 0.3

    benchmark_to_mem = {'650.vid': 2048, '660.map-reduce': 256, '670.auth':256, '680.excamera': 256, '690.ml': 1024, '6100.1000-genome': 2048}

    for idx, platform in enumerate(args.platforms):
        ys = []
        es = []
        print(benchmark_to_mem)
        for benchmark in benchmark_to_mem.keys():
            m = benchmark_to_mem[benchmark]
            filename = f"{args.experiment}_{m}_processed.csv" #if platform != "azure" else f"{args.experiment}_processed.csv"
            path = os.path.join("./../perf-cost", benchmark, platform + "/2024", filename)
            if not os.path.exists(path):
                print(path)
                continue

            df = read(path)

            cold = 100*df["is_cold"]
            y = cold.mean()
            print("benchmark: ", benchmark, "platform: ", platform, "% cold: ", y)
        es = np.asarray(es).transpose()

        name = platform_names[platform]
        o = ((len(args.platforms)-1)*w)/2.0 - idx*w


if __name__ == "__main__":
    bar_plot()
