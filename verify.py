import os
import glob
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exclude", nargs='+', default=[])
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

    if benchmark in num_rows_per_invo:
        exp_cnt = num_rows_per_invo[benchmark]
        if exp_cnt != cnt[0]:
            raise RuntimeError(f"Number of measurments for {benchmark} don't match: Expected {exp_cnt}, got {cnt[0]}.")
    else:
        num_rows_per_invo[benchmark] = cnt[0]

