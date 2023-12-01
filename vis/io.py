import os
import glob

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if __name__ == "__main__":
    #for f in glob.glob("perf-cost/*/aws_io/*.csv"):
    for f in glob.glob("./../perf-cost/*/aws_io/*.csv"):
        print(f)
        _, _, _, benchmark, _, filename = f.split("/")

        df = pd.read_csv(f)
        invos = df.groupby("request_id").agg({"blob.download": sum, "blob.upload": sum})
        assert(np.std(invos["blob.download"]) == 0)
        assert(np.std(invos["blob.upload"]) == 0)

        upload_mb = np.mean(invos["blob.upload"]) / 1024**2
        download_mb = np.mean(invos["blob.download"]) / 1024**2

        print(benchmark, "upload:", upload_mb, "download:", download_mb)
