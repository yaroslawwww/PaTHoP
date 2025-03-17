# coding: utf-8
import numpy as np
import sys
from Research import *
from matplotlib import pyplot as plt
import concurrent.futures



def process_epsilon(deviation, shares, size, divisor):
    result = parallel_research(r_values=[28,28, 28 + deviation],
                               ts_size=(np.array([200000] + list(np.array(shares) * size * divisor))).astype(np.uint64),
                               gap_number=1000,
                               test_size_constant=50)
    return result[0], result[1], result[2]




def main():
    divisor = 10
    shares = [1,0]
    size = int(float(sys.argv[1]))
    deviation = 0

    rmse, np_points , mean_affiliation = process_epsilon(deviation, shares, size, divisor)
    with open(f"/home/ikvasilev/fast_epsilon_counter/rmses_np_points_affiliation_size.txt", "a") as f:
        f.write(str(size) + "," + str(rmse) + "," + str(np_points) + "," + str(mean_affiliation) + "\n")


if __name__ == '__main__':
    main()