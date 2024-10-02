We provide an overview of how each of the scripts can be used to reproduce the plots and analyses in the paper.

## Cold Starts

Compute the percent of cold starts for a given benchmark in a specific measurement mode (Table 5). 

```
python cold_starts.py -p gcp aws azure -e burst
```

## Cold vs Warm Starts

Plot the critical path and overhead of cold vs warm workflow executions for a given benchmark (Figure 11). 

```
python overhead-cold-vs-warm.py -p aws/2024 gcp/2024 azure/2024 -b 660.map-reduce -m 256 -e burst
```

## Invocation Latency

Plot the invocation latency based on the measurements from the function chain micro-benchmark (Fig. 8b).

```
python invo_lat.py -e warm
```

## OS Noise

Plot the relative suspension time based on the measurements from the selfish detour micro-benchmark (Fig. 12a).

```
python os_noise.py -e warm
```

## Overhead

Plot the critical path and overhead of a given benchmark in a specific measurement mode and with a specific memory configuration (Fig. 7).

```
python overhead.py -b  6200.trip-booking -e burst -m 128 -p gcp/2024 aws/2024 azure/2024
```

## Overhead-2022-vs-2024

Plot critical path and overhead of a given benchmark for different executions to compare, e.g., between years, for a specific measurement mode and with a specific memory configuration (Fig. 15).

```
python overhead-2022-vs-2024.py -p gcp/2024 aws/2024 azure/2024 gcp/2022 aws/2022 azure/2022 -e burst -b 660.map-reduce -m 256
```


Plot normalized vs. original critical path of a given benchmark in a specific measurement mode and with a specific memory configuration (Fig. 12b, c).

```
python overhead-2022-vs-2024.py -p gcp/2024 aws/2024 azure/2024 -e burst -b 690.ml -m 1024 --noise --no-overhead
```


## Overhead Parallel

Plot the heatmap of the overhead incurred during parallel executions as measured by the parallel sleep micro-benchmark (Fig. 9).

```
python overhead_parallel.py
```

## Overhead Storage

Plot the overhead incurred by storage I/O as measured by the parallel download micro-benchmark (Fig. 8a). 

```
python overhead_storage.py -e burst
```

## Plot Scaling

Plot complete runtime of 1000Genomes workflow on cloud platforms and HPC system Ault (Fig. 13a):

```
python plot-scaling.py --complete_runtime
```

Plot scaling of the runtime of the individuals task of the 1000Genomes workflow on cloud platforms and HPC system Ault (Fig. 13b):

```
python plot-scaling.py
```

## Pricing

Plot pricing of a given benchmark in a specific measurement mode and with a specific memory configuration (Fig. 14):

```
python3 pricing.py -p gcp/2024 aws/2024 azure/2024 -b 680.excamera -m 256 -e burst
```

## Runtime

```
python runtime.py --benchmark 6200.trip-booking --visualization violin -p gcp/2024 aws/2024 azure/2024 -m 128 -e burst
```

## Confidence Intervals

Print non-parametric confidence intervals of each repetition of 30 concurrent invocations of a given benchmark in a specific measurement mode and with a specific memory configuration (Sec. 7A). 

```
python confidence-intervals.py -b 660.map-reduce -m 256 -p  gcp/2024 -e burst
```

## Scalability

Plot the distinct number of containers used for 30 consecutive worklow invocations (Fig. 10).

```
python scalability.py --benchmark 6200.trip-booking -p gcp/2024 aws/2024 azure/2024 -m 128 -e burst
```

