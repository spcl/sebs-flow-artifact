import pandas as pd
import scipy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--platform") #, nargs='+', default=["aws", "azure", "gcp"])
parser.add_argument("-t", "--test")
args = parser.parse_args()

#read in data
benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-30-reps-6", "burst_256_processed.csv")
df_2 = pd.read_csv(benchmark_path)
#print(benchmark_path)

benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-32-reps-3", "burst_256_processed.csv")
df_3_reps = pd.read_csv(benchmark_path)

'''
benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-32-reps-1", "burst_256_processed.csv")
df_1 = pd.read_csv(benchmark_path)
df_1["times"] = df_1["end"] - df_1["start"]
benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-32-reps-3", "burst_256_processed.csv")
df_2 = pd.read_csv(benchmark_path)

if args.platform == "gcp":
    benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-32-rep-0-date-7-1", "burst_256_processed.csv")
else:
    benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-32-rep0-date-7-1", "burst_256_processed.csv")
df_3 = pd.read_csv(benchmark_path)
df_3["times"] = df_3["end"] - df_3["start"]

if args.platform == "gcp":
    benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-32-rep-0", "burst_256_processed.csv")
else: 
    benchmark_path = os.path.join("../perf-cost", "660.map-reduce", args.platform, "batch-size-32-rep0-date-8-1", "burst_256_processed.csv")
df_4 = pd.read_csv(benchmark_path)
df_4["times"] = df_4["end"] - df_4["start"]
'''

df_2["times"] = df_2["end"] - df_2["start"]
df_2.sort_values('start')
print(len(df_2))
df_2_1 = df_2.loc[0:299]
df_2_2 = df_2.loc[300:599]
df_2_3 = df_2.loc[600:899]
df_2_4 = df_2.loc[900:1199]
df_2_5 = df_2.loc[1200:1499]
df_2_6 = df_2.loc[1500:1799]

def compute_kl(x, y, no_buckets):
    #split into buckets
    #it is just times. 

    min_val = x.min()
    max_val = x.max()
    if y.min() < min_val:
      min_val = y.min()
    if y.max() > max_val:
      max_val = y.max()
    dist = max_val - min_val
    
    #now compute brackets for buckets
    #no_buckets = 10
    incr_size = float(dist / no_buckets)
    frequencies_x = []
    frequencies_y = []
    for i in range(0,no_buckets):
        #iterate over values and put them in buckets.
      start = min_val + (i * incr_size)
      end = start + incr_size
      
      freq_x = 0
      if i == no_buckets-1:
        freq_x = len(x.loc[(x >= start)])
      else:
        freq_x = len(x.loc[(x >= start) & (x < end)])
      frequencies_x.append(freq_x)
      
      freq_y = 0
      if i == no_buckets-1:
        freq_y = len(y.loc[(y >= start)])
      else:
        freq_y = len(y.loc[(y >= start) & (y < end)])
      frequencies_y.append(freq_y)
    #now compute probabilities out of the frequencies
    probs_x = []
    for elem in frequencies_x: 
      if elem == 0:
        probs_x.append(0.0000000000000000000000000000000000000000001)
      else:
        prob = float(elem / len(x))
        probs_x.append(prob)
    probs_y = []
    for elem in frequencies_y:
      if elem == 0:
        probs_y.append(0.0000000000000000000000000000000000000000001)
      else: 
        prob = float(elem / len(y))
        probs_y.append(prob)
        
    return scipy.special.rel_entr(probs_x, probs_y)

def compute_total_runtime():
    
    funcs_df_1 = pd.concat([df_2_1, df_2_2, df_2_3]) #, df_2_4, df_2_5, df_2_6])
    #funcs_df_1 = df_2_1.groupby("func")

    #funcs_df_2 = df_2_3
    # now assemble dfs for testing.
    #funcs_df_1 = pd.concat([df_2_1])
    #funcs_df_1 = df_2_1.groupby("func")

    funcs_df_2 = df_2_4

    invos_df_1 = funcs_df_1.groupby("request_id")
    my_df_1 = invos_df_1["end"].max() - invos_df_1["start"].min()
    invos_df_2 = funcs_df_2.groupby("request_id")
    my_df_2 = invos_df_2["end"].max() - invos_df_2["start"].min()    
    
    if args.test == 'kl':
      #kl divergence test
      for i in range(10,30,5):
        result = compute_kl(my_df_1, my_df_2, i)
      #result = scipy.special.kl_div(my_df_1, my_df_2)
        print("Buckets=", i , round(sum(result),2))
      
    else: 
      #KS and Man Whitney U Test
      result_ks_2samp = scipy.stats.ks_2samp(my_df_1, my_df_2, alternative='two-sided', method='auto')
      result, p = scipy.stats.mannwhitneyu(my_df_1, my_df_2, alternative='two-sided')
      print("p Smirnov", result_ks_2samp.pvalue, "   p mannwhitneyu", p)

def compute_per_function():
    # now assemble dfs for testing.
    funcs_df_1 = pd.concat([df_2_1, df_2_2, df_2_3, df_2_4, df_2_5, df_2_6]).groupby("func")
    #funcs_df_1 = df_2_1.groupby("func")

    funcs_df_2 = df_3_reps.groupby("func")

    for key, item in funcs_df_1:
            #print("key: ", key)
            my_df_1 = funcs_df_1.get_group(key)["times"]
            my_df_2 = funcs_df_2.get_group(key)["times"]
            result_ks_2samp = scipy.stats.ks_2samp(my_df_1, my_df_2, alternative='two-sided', method='auto')
            result, p = scipy.stats.mannwhitneyu(my_df_1, my_df_2, alternative='two-sided')
            #print(result)
            print("func: ", key, "    p mannwhitneyu", p, "   p Smirnov", result_ks_2samp.pvalue)
            # the p-value is lower than our threshold of 0.05, so we reject the null hypothesis in favor of the default “two-sided” alternative. this means: p should be greater than 0.05!
            # alternative 2samp: the data were not drawn from the same distribution.
            # alternative mannwhitneyu: the distributions are not equal. 
            # --> p-value should be >0.05
        
if __name__ == "__main__":
    #compute_per_function()
    compute_total_runtime()
