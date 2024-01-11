import json 

f = open('/home/larissa/Serverless/benchmark-workflows/master-serverless-benchmarks/serverless-benchmarks/config/example.json')

data = json.load(f)

threads = [2, 4, 8, 16]
duration = [1, 5, 10, 15, 20]

#change platform
#data["deployment"]["name"] = "gcp"
data["deployment"]["gcp"]["region"] = "us-east1"

for t in threads:
    for d in duration:
        #now change input-size
        perf_cost_experiment = data["experiments"]["perf-cost"]
        perf_cost_experiment["input-size"] = str(t) + "-" + str(d)
        

        filename = '/home/larissa/Serverless/benchmark-workflows/master-serverless-benchmarks/serverless-benchmarks/config/configs-parallel-sleep/' + str(t) + "-" + str(d) + ".json"

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

