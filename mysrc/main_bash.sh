#!/bin/bash

# Параметры
start=0
end=30000
steps=24

experiment_name="Size_experiment"



# Вычисляем шаг
step=$(( (end - start) / (steps - 1) ))

for (( i=0; i<steps; i++ )); do
    size=$(( start + i * step ))
    echo "Submitting job with size=$size"
    sbatch -A proj_1716 ./bash 0.1 10 $size $experiment_name 5000
done

# Параметры
start=0
end=30000
steps=24


# Вычисляем шаг
step=$(( (end - start) / (steps - 1) ))
for (( i=0; i<steps; i++ )); do
    size=$(( start + i * step ))
    echo "Submitting job with size=$size"
    sbatch -A proj_1716 ./bash 0.01 10 $size $experiment_name 5000
done
# Параметры
start=0
end=30000
steps=24


# Вычисляем шаг
step=$(( (end - start) / (steps - 1) ))
for (( i=0; i<steps; i++ )); do
    size=$(( start + i * step ))
    echo "Submitting job with size=$size"
    sbatch -A proj_1716 ./bash 0.0001 10 $size $experiment_name 5000
done

# Параметры
start=0
end=30000
steps=24


# Вычисляем шаг
step=$(( (end - start) / (steps - 1) ))
for (( i=0; i<steps; i++ )); do
    size=$(( start + i * step ))
    echo "Submitting job with size=$size"
    sbatch -A proj_1716 ./bash 2 10 $size $experiment_name 5000
done
# Параметры
start=0
end=30000
steps=24


# Вычисляем шаг
step=$(( (end - start) / (steps - 1) ))
for (( i=0; i<steps; i++ )); do
    size=$(( start + i * step ))
    echo "Submitting job with size=$size"
    sbatch -A proj_1716 ./bash 10 10 $size $experiment_name 5000
done

# Параметры
start=5000
end=30000
steps=24


# Вычисляем шаг
step=$(( (end - start) / (steps - 1) ))
for (( i=0; i<steps; i++ )); do
    size=$(( start + i * step ))
    echo "Submitting job with size=$size"
    sbatch -A proj_1716 ./bash 0 10 0 $experiment_name $size
done