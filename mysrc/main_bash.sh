#!/bin/bash
#SBATCH --job-name=launcher
#SBATCH --output=launcher_%j.out
#SBATCH --error=launcher_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

#DEVIATIONS=("0.1"  "2" "10" "-0.1" "0.01" "-0.01" "0.0001" "3" "4" "5" "6" "7" "8" "9")
##DEVIATIONS=("0.1")
#for deviation in "${DEVIATIONS[@]}"; do
#  for i in $(seq 0 23); do
#    added_size=$((0 + (i * 30000) / 23))
#
#    sbatch -A proj_1716 ./bash \
#      "$deviation" \
#      "10" \
#      "$added_size" \
#      "Daemon" \
#      "10000"
#  done
#done

for i in $(seq 0 23); do
  base_size=$((10000 + (i * 30000) / 23))

  sbatch -A proj_1716 ./bash \
    "0" \
    "10" \
    "0" \
    "Daemon" \
    "$base_size"
done