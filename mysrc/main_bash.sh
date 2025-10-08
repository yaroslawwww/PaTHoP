#!/bin/bash

DEVIATIONS=("0.1"  "2" "10" "-0.1" "0.01" "-0.01" "0.0001")
for deviation in "${DEVIATIONS[@]}"; do
  for i in $(seq 0 23); do
    added_size=$((0 + (i * 30000) / 23))

    sbatch -A proj_1716 ./bash \
      "$deviation" \
      "10" \
      "$added_size" \
      "Daemon" \
      "10000"
  done
done

for i in $(seq 0 23); do
  base_size=$((10000 + (i * 30000) / 23))

  sbatch -A proj_1716 ./bash \
    "0" \
    "10" \
    "0" \
    "Daemon" \
    "$base_size"
done