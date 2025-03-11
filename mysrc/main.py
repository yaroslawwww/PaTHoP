# coding: utf-8
import sys

import numpy as np

from Research import *
from matplotlib import pyplot as plt
import concurrent.futures



def process_epsilon(deviation, shares, size, divisor):
    result = parallel_research(r_values=[28, 28 + deviation],
                               ts_size=(np.array(shares) * size * divisor).astype(np.uint64),
                               gap_number=3,
                               test_size_constant=100)
    return deviation, result[0], result[1], result[2]


def main():
    divisor = 10
    shares = [0.5,0.5]
    size = 250000
    deviation = float(sys.argv[1])

    deviation, rmse, np_points, mean_affiliation = process_epsilon(deviation, shares, size, divisor)
    print(str(deviation) + "," + str(rmse) + "," + str(np_points) + "," + str(mean_affiliation) + "\n")
    with open(f"/home/ikvasilev/fast_epsilon_counter/ihateyou.txt", "a") as f:
        f.write(str(deviation) + "," + str(rmse) + "," + str(np_points) + "," + str(mean_affiliation) + "\n")


if __name__ == '__main__':
    main()
