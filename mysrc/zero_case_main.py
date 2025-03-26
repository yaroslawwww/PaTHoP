# coding: utf-8
import numpy as np
import sys
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures



def process_deviation(deviation, shares, size,test_size):
    result = parallel_research(r_values=[28,28, 28 + deviation],
                               ts_size=(np.array([20000] + list(np.array(shares) * size))).astype(np.uint64),
                               gap_number=1000,
                               test_size_constant=test_size)
    return result[0], result[1], result[2]




def main():
    shares = [1,0]
    size = 25000
    deviation = 0
    test_size = float(sys.argv[2])
    rmses, np_points, mean_affiliation = process_deviation(deviation, shares, size,test_size)
    with open(f"/home/ikvasilev/fast_epsilon_counter/rmse_and_np.txt", "a") as f:
        f.write(str(deviation) + "," + str(test_size) + "," + str(size) + "," + str(rmses) + "," + str(np_points) + "," + str(mean_affiliation) + "\n")


if __name__ == '__main__':
    main()