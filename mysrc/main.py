# coding: utf-8
import sys

import numpy as np
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures


def process_epsilon(epsilon, shares, size, divisor):
    # Каждый процесс выполняет вызов threaded_research с заданным epsilon
    result = threaded_research(r_values=[28, 28 + epsilon],
                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                               gap_number=2,
                               test_size_constant=50)
    return epsilon, result[0], result[1], result[2]


def main():
    divisor = 10
    base_shares = [0.5, 0.5]
    size = 250000
    # Параллелизация цикла по epsilon с использованием процессов
    epsilon = sys.argv[1]
    rmses = []
    np_points = []
    affiliation_list = []

    eps, rmse, np_point, affiliation_array = process_epsilon(epsilon, base_shares, size, divisor)
    rmses.append(rmse)
    np_points.append(np_point)
    affiliation_list.append(affiliation_array)
    with open(f"/home/ikvasilev/fast_epsilon_counter/fast_epsilon{epsilon}.txt", "a") as f:
        f.write(
            str(epsilon) + "," + str(np_point) + "," + str(rmse) + "," + str(affiliation_list) + "\n")


if __name__ == '__main__':
    main()
