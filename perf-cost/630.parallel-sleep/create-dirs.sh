#!/bin/bash

threads=( 2 4 8 16)
durations=( 1 5 10 15 20)

echo bla

for thread in "${threads[@]}"; do
  for duration in "${durations[@]}"; do
    #mkdir gcp_${thread}t_${duration}s
    #mv aws_${thread}t_${duration}s/630.parallel-sleep/aws/* aws_${thread}t_${duration}s
    rm -r aws_${thread}t_${duration}s/630.parallel-sleep
  done
done
