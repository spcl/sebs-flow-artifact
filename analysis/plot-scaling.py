import pandas as pd
import os
import seaborn as sb
import matplotlib.pyplot as plt
import scipy

platform_names = {
    "aws": "AWS",
    "gcp": "Google Cloud",
    "azure": "Azure"
}

def print_statistics(df):
    invos = df.groupby(["platform", "jobs"])
    print(invos["duration"].mean())
    for key, item in invos:
        my_df = invos.get_group(key)
        print(key, "variation: ", round(scipy.stats.variation(my_df["duration"]),3))

def plot_average_runtime(df):
    invos = df.groupby("request_id")
    d_total = invos["end"].max() - invos["start"].min()

    rows_list = []
    for key, item in invos:
        my_df = invos.get_group(key)
        d_total = my_df["duration"].mean()
        dict1 = {'jobs': my_df['jobs'].min(), 'duration': d_total, 'platform': my_df['platform'].min()}
        rows_list.append(dict1)

    df = pd.DataFrame(rows_list)

    g = sb.catplot(data=df, kind="bar", x="jobs", y="duration", hue="platform", errorbar="sd", legend_out=False, height=6, aspect=3.8/3)
    g.despine(left=True)

    

    plt.legend(fontsize=20)
    g.set_axis_labels("Number of individuals jobs", "Duration [s]", fontsize=20)
    g.legend.set_title("")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.show()
    


def plot_complete_runtime(df): 
    invos = df.groupby("request_id")
    d_total = invos["end"].max() - invos["start"].min()

    rows_list = []
    for key, item in invos:
        my_df = invos.get_group(key)
        if my_df['platform'].min() == "Ault":
          #find out per function what the longest-running was
          funcs = my_df.groupby("phase")
          d_total = 0
          for key2, item2 in funcs:
              func_df = funcs.get_group(key2)
              d_total += func_df['duration'].max()
        else:
          d_total = my_df["end"].max() - my_df["start"].min()
        dict1 = {'jobs': my_df['jobs'].min(), 'duration': d_total, 'platform': my_df['platform'].min()}
        rows_list.append(dict1)
        

    df = pd.DataFrame(rows_list)

    g = sb.catplot(data=df, kind="bar", x="jobs", y="duration", hue="platform", errorbar="sd", legend_out=False, height=6, aspect=3.8/2.8)
    g.despine(left=True)
    
    #annotate plot
    medians = df.groupby(['jobs', 'platform'])['duration'].median()
    medians = medians.round(2)
    vertical_offset = df['duration'].median() * 0.5# offset from median for display
    
    for container in g.ax.containers:
        g.ax.bar_label(container, fmt='%.2f', padding=2,fontsize=20)
    
    plt.legend(fontsize=20)
    g.set_axis_labels("Number of individuals jobs", "Duration [s]", fontsize=20)
    g.legend.set_title("")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    print_statistics(df)

def read_csv(platform, jobs):
    if jobs == 5:
        df = pd.read_csv(os.path.join("../perf-cost", "6100.1000-genome", platform, "2024", "burst_2048_processed.csv"))
    else:
        df = pd.read_csv(os.path.join("../perf-cost", "6101.1000-genome-individuals-" + str(jobs), platform, "burst_2048_processed.csv"))
    df['platform'] = platform_names[platform]
    df['jobs'] = jobs
    df['duration'] = df['end'] - df['start']
    return df

colors = sb.color_palette("colorblind")
sb.set_theme()
sb.set_context("paper")

five_jobs = []
ten_jobs = []
twenty_jobs = []

#first read input files from ault and find out mean values + CoV
input_ault = "../1000genomes-ault/1250lines"
for file in os.listdir(input_ault):
    _, jobs, rep = file.split("-")
    df = pd.read_csv(os.path.join(input_ault, file), sep=",", names=["func","duration"], header=None)
    df['request_id'] = rep + "-" + jobs
    df['platform'] = 'Ault'
    df['duration'] = df['duration'] / 60
    if int(jobs) == 5:
      five_jobs.append(df)
    elif int(jobs) == 10:
      ten_jobs.append(df)
    else:
      twenty_jobs.append(df)

df_five_jobs = pd.concat(five_jobs)
df_five_jobs['jobs'] = 5
df_ten_jobs = pd.concat(ten_jobs)
df_ten_jobs['jobs'] = 10
df_twenty_jobs = pd.concat(twenty_jobs)
df_twenty_jobs['jobs'] = 20

dfs = [df_five_jobs, df_ten_jobs, df_twenty_jobs]

jobs = [5, 10, 20]
for job in jobs:
    df_aws = read_csv('aws', job)
    dfs.append(df_aws)
    df_gcp = read_csv('gcp', job)
    dfs.append(df_gcp)
    df_azure = read_csv('azure', job)
    dfs.append(df_azure)
    
df = pd.concat(dfs)

df.loc[df.func == "individuals", "phase"] = "phase0"
df.loc[df.func == "individuals_merge", "phase"] = "phase1"
df.loc[df.func == "sifting", "phase"] = "phase1"
df.loc[df.func == "frequency", "phase"] = "phase2"
df.loc[df.func == "mutation_overlap", "phase"] = "phase2"


#df_individuals = df.loc[df.func == "individuals"]
#plot_average_runtime(df_individuals)

df_5_jobs = df.loc[df.jobs == 5]
plot_complete_runtime(df_5_jobs)
