import os
import glob
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exclude", nargs='+', default=[])
parser.add_argument("--processed-head", type=int, default=None)
args = parser.parse_args()
num_rows_per_invo = dict()

for f in glob.glob("perf-cost/*/*/*.csv"):
    print(f"Checking {f}...")
    _, benchmark, platform, experiment = f.split("/")
    if benchmark in args.exclude:
        print(f"Skipped.")
        continue

    df = pd.read_csv(f)

    invos = df.groupby("request_id")
    cnt = invos.size()
    assert(np.all(cnt == cnt[0]))

    funcs = invos["func"].value_counts()
    summary = ""
    for func, count in funcs[funcs.index.levels[0][0]].items():
        summary += f"{func}: {count}, "

    print(f"{cnt[0]} measurments/invocation: {summary[:-2]}")

    cnt = cnt[0]
    if "azure" in platform:
        if benchmark in ["650.vid", "660.map-reduce", "670.auth", "680.excamera", "690.ml"]:
            cnt -= 1

    if benchmark in num_rows_per_invo:
        exp_cnt = num_rows_per_invo[benchmark]
        if exp_cnt != cnt:
            raise RuntimeError(f"Number of measurments for {benchmark} don't match: Expected {exp_cnt}, got {cnt}.")
    else:
        num_rows_per_invo[benchmark] = cnt


if args.processed_head:
    print(f"Clipping processed files to {args.processed_head} invocations")

    for f in glob.glob("perf-cost/*/*/*_processed.csv"):
        print(f"Checking {f}...")

        df = pd.read_csv(f)
        invos = df.groupby("request_id")
        valid_invos = invos.filter(lambda g: np.all((g["provider.execution_duration"] > 0) | (g["provider.request_id"].isnull())))
        req_ids = valid_invos["request_id"].unique()

        if len(req_ids) < args.processed_head:
            print(f"Insufficient valid invocations for {f}: {len(req_ids)}/{args.processed_head}")
            print("Skipping...")
            continue
        elif len(req_ids) == args.processed_head:
            continue

        df = df.loc[df["request_id"].isin(req_ids[:args.processed_head])]

        columns = [c for c in df.columns if "Unnamed" not in c]
        df.to_csv(f, columns=columns, index=False)

