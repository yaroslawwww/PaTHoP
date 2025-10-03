#!/bin/bash

DEVIATIONS=("0.1" "0.01" "0.0001" "2" "10" "-0.01" "-0.1")
DAEMONS=("basic_dbscan" "spread" "iqr" "entropy")

for daemon in "${DAEMONS[@]}"; do
  for deviation in "${DEVIATIONS[@]}"; do
    for i in $(seq 0 23); do
      added_size=$((0 + (i * 30000) / 23))

      sbatch submit_job.sh \
        "$deviation" \
        "10" \
        "$added_size" \
        "Daemon" \
        "10000" \
        "$daemon"
    done
  done
done

for daemon in "${DAEMONS[@]}"; do
    for i in $(seq 0 23); do
      base_size=$((10000 + (i * 30000) / 23))

      sbatch submit_job.sh \
        "$deviation" \
        "10" \
        "0" \
        "Daemon" \
        "$base_size" \
        "$daemon"
    done
done